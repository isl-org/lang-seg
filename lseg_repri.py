from collections import defaultdict
import itertools
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.lseg_module_zs import LSegModuleZS
from additional_utils.models import LSeg_MultiEvalModule
from fewshot_data.common.logger import Logger# AverageMeter
from fewshot_data.common.vis import Visualizer
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.common import utils
from fewshot_data.data.dataset import FSSDataset
from repri_classifier import Classifier, batch_intersectionAndUnionGPU, to_one_hot
from types import SimpleNamespace
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_resnet101",
            help="backbone name (default: clip_resnet101)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="pascal",
            help="dataset name (default: pascal)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=2,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=2,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False if torch.cuda.is_available() else True, #True,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default='./checkpoints/pascal_fold0.ckpt', help="checkpoint to test"
            # NOTE: default is teh pascal_fold0 trained weight, which mean it is trained on fold 1,2,3. 
            # Refer to the LSeg github repo for more details
        )
        # evaluation option
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )

        parser.add_argument(
            "--module",
            default='',
            help="select model definition",
        )

        # test option
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )

        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )

        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
        )

        parser.add_argument(
            "--use_pretrained",
            type=str,
            default="True",
            help="whether use the default model to intialize the model",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        # fewshot options
        parser.add_argument(
            '--nshot', 
            type=int, 
            default=3
            )
        parser.add_argument(
            '--fold', 
            type=int, 
            default=0, 
            choices=[0, 1, 2, 3]
            )
        parser.add_argument(
            '--nworker', 
            type=int, 
            default=0
            )
        parser.add_argument(
            '--bsz', 
            type=int, 
            default=1 # NOTE: always = 1
            )
        parser.add_argument(
            '--benchmark', 
            type=str, 
            default='pascal',
            choices=['pascal', 'coco', 'fss', 'c2p']
            )
        parser.add_argument(
            '--datapath', 
            type=str, 
            default='/Users/maxxyouu/Desktop/lang-seg-fork/data/'
            )

        parser.add_argument(
            "--activation",
            choices=['relu', 'lrelu', 'tanh'],
            default="relu",
            help="use which activation to activate the block",
        )


        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args
    

def episodic_validate(args):
    print('==> Start testing')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model = args.module.net.eval().cuda()
    else:
        model = args.module.net.eval().cpu()

    # total number of test cases / batch size = number of episode, a episode contains a multiple (query, supports) pairs of random classes
    nb_episodes = int(args.test_num / args.batch_size_val)

    # ========== Metrics initialization  ==========    
    H, W = args.image_size, args.image_size
    # c = model.module.bottleneck_dim
    # get the feature size TODO: COMPLETED
    # h = model.module.feature_res[0]
    # w = model.module.feature_res[1]
    c = 512
    h = 240
    w = 240

    # intialize the container to store the results
    runtimes = torch.zeros(args.n_runs)
    deltas_init = torch.zeros((args.n_runs, nb_episodes, args.batch_size_val))
    deltas_final = torch.zeros((args.n_runs, nb_episodes, args.batch_size_val))
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ========== Perform the runs  ==========
    # mainly to repeat the same set of experiment n_run times
    for run in tqdm(range(args.n_runs)):

        # =============== Initialize the metric dictionaries ===============
        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        IoU = defaultdict(int)

        # =============== episode = group of tasks ===============
        # for each episode, it contains multiple (query, supports) pairs, each pair requires its own \theta parameters of the classifier 
        # to make inference on the pair corresponding query image.

        for e in tqdm(range(nb_episodes)):

            # NOTE: args.batch_size_val is equivalent to say number of pairs/tasks in a batch
            features_s = torch.zeros(args.batch_size_val, args.shot, c, h, w).to(device)
            text_s = torch.zeros(args.batch_size_val, 1, c).to(device) # dim[1] is the label text
            features_q = torch.zeros(args.batch_size_val, 1, c, h, w).to(device)
            gt_s = 255 * torch.ones(args.batch_size_val, args.shot, args.image_size,
                                    args.image_size).long().to(device)
            gt_q = 255 * torch.ones(args.batch_size_val, 1, args.image_size,
                                    args.image_size).long().to(device)
            n_shots = torch.zeros(args.batch_size_val).to(device)
            classes = []  # All classes considered in the tasks

            # =========== Generate tasks and extract features for each task - a pair of query and supports ===============
            with torch.no_grad():
                for i in range(args.batch_size_val):
                    # load each pair in the episode one by one
                    batch = next(args.val_loader)
                    qry_img = batch['query_img']
                    q_label = batch['query_mask']
                    spprt_imgs = batch['support_imgs']
                    s_label = batch['support_masks']
                    subcls = batch['class_id']
                    iter_num += 1

                    # place it to the corresponding gpu/cpu/the ith gpu in the cluster
                    q_label = q_label.to(device)
                    spprt_imgs = spprt_imgs.to(device)
                    s_label = s_label.to(device)
                    qry_img = qry_img.to(device)

                    # get the final feature tensor of the support images and the query image
                    # TODO: COMPLETED
                    # f_s = model.module.extract_features(spprt_imgs.squeeze(0)) #[shots, c, h, w]
                    # f_q = model.module.extract_features(qry_img) #[1, c, h, w]

                    # TODO: merge the text and support image features
                    f_s, t_s, _ = model.extract_features(spprt_imgs.squeeze(0), subcls)
                    f_q, _, _ = model.extract_features(qry_img, subcls)

                    shot = f_s.size(0)
                    n_shots[i] = shot
                    features_s[i, :shot] = f_s.detach() # add the feature tensor of the shots to the container for each pair in the batch
                    features_q[i] = f_q.detach() # same for the query but only one shot here
                    t_s = t_s[0]
                    text_s[i] = t_s.detach()

                    # store the corresponding labels
                    gt_s[i, :shot] = s_label
                    gt_q[i, 0] = q_label
                    
                    # add individual class label in a batch to the container, recall item() only work for tensor that contains one element only
                    classes.append([class_.item() for class_ in subcls])
            
            # =========== Normalize features along channel dimension ===============
            # if args.norm_feat:
            #     features_s = F.normalize(features_s, dim=2)
            #     features_q = F.normalize(features_q, dim=2)

            # =========== Create a callback is args.visdom_port != -1 ===============
            # callback = VisdomLogger(port=args.visdom_port) if use_callback else None
            callback = None

            # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
            classifier = Classifier(args)
            classifier.init_prototypes(features_s, features_q, text_s, gt_s, gt_q, classes, callback)
            batch_deltas = classifier.compute_FB_param(features_q, gt_q)
            deltas_init[run, e, :] = batch_deltas.cpu()

            # =========== Perform RePRI inference ===============

            # train the one layer classifier to learn the optimal average prototype in the support images
            batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback)
            deltas_final[run, e, :] = batch_deltas

            # perform actual inference on the query image with the prototype learnt from the support shots
            logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w]
            logits = F.interpolate(logits,
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=True) # upsample the logit score to the original image size
            probas = classifier.get_probas(logits).detach() # get the probabilty at each spatial location [n_tasks, shot, num_classes, h, w]
            intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2)  # [n_tasks, shot, num_class]
            intersection, union = intersection.cpu(), union.cpu()

            # ================== Log metrics ==================
            one_hot_gt = to_one_hot(gt_q, 2)
            valid_pixels = gt_q != 255
            loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
            loss_meter.update(loss.item())
            for i, task_classes in enumerate(classes):
                for j, class_ in enumerate(task_classes):
                    cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                    cls_union[class_] += union[i, 0, j + 1]

            for class_ in cls_union:
                IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)

            if (iter_num % 5 == 0):
                mIoU = np.mean([IoU[i] for i in IoU])
                print('Test: [{}/{}] '
                    'mIoU {:.4f} '
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
                                                                            args.test_num,
                                                                            mIoU,
                                                                            loss_meter=loss_meter,
                                                                            ))

        mIoU = np.mean(list(IoU.values()))
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()

def load_checkpoint(module_def):
    return module_def.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.datapath,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=args.arch_option,
        use_pretrained=args.use_pretrained,
        strict=args.strict,
        logpath='fewshot/logpath_4T/',
        fold=args.fold,
        block_depth=0,
        nshot=args.nshot,
        finetune_mode=False,
        activation=args.activation,
    )

def test(args):
    assert(args.backbone == 'clip_resnet101')
    module_def = LSegModuleZS
    module = load_checkpoint(module_def)

    Evaluator.initialize()
    image_size = 480
    # pascal and imagenet use the same norm and mean
    FSSDataset.initialize(img_size=image_size, datapath=args.datapath, use_original_imgsize=False, imagenet_norm=True)

    # dataloader
    args.benchmark = args.dataset
    dataloader, dataset = FSSDataset.build_dataloader(
        benchmark=args.benchmark, 
        bsz=args.bsz, 
        nworker=args.nworker, 
        fold=args.fold, 
        split='test', 
        shot=args.nshot)
    
    # TODO: 
    # 1. try one-shot with text embedding and without text embedding; 
    # 2. try different combination of adapt_iter, lr, and param updates (the frequency); 
    # 3. try the same set of hyperparameters with oracle
    # 4. try with and without the shannon entropy and/or KL divergence
    params = {
        'module': module,
        'image_size':  image_size,
        'test_num': len(dataset.img_metadata), # total number of test cases
        'batch_size_val': 10, # NOTE: this is different than the args.bsz
        'n_runs': 1, # repeat the experiment 1 time
        'shot': args.nshot,
        'val_loader': iter(dataloader),
        'benchmark': args.dataset, # pascal
        # the following are for the RePRI classifier
        'temperature': 20,
        'adapt_iter': 50,
        'weights': [1.0, 'auto', 'auto'],
        'cls_lr': 0.02,
        'FB_param_update': [10, 20, 30, 40],
        'cls_visdom_freq': 5, # might not need it
        # NOTE: we only need the following when it is a oracle experiment
        'FB_param_type': 'joe', 
        'FB_param_noise': 0
    }

    # CLASSIFIER:
    #   distance: cos
    #   temperature: 20.
    #   adapt_iter: 50
    #   FB_param_type: soft
    #   weights: [1.0, 'auto', 'auto']  #  [0.5, 1.0, 0.1]
    #   cls_lr: 0.025
    #   FB_param_update: [10]
    #   cls_visdom_freq: 5

    # RePRI inference
    episodic_validate(SimpleNamespace(**params))

def hyperparameter_tuning():
    shots = [1, 2, 5, 10, 20]
    temperatures = [0.1, 0.5, 2, 8, 16, 20, 40]
    iterations = [5, 10, 20, 30, 40, 50, 60, 70, 80]
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
    fb_params = [[10], [20]] # TODO:
    fb_params_types = ['joe', 'oracle']
    with_text_embeddings = [True, False]

    hy_params = itertools.product(shots, temperatures, iterations, learning_rates, fb_params, fb_params_types, with_text_embeddings)
    for shot, tmp, iter, lr, fb_updates, fb_type, with_t in hy_params:
        args = Options().parse()
        torch.manual_seed(args.seed)
        
        # classifier hyper parameter changing
        args.nshot = shot
        args.temp = tmp
        args.adapt_iter = iter
        args.cls_lr = lr
        args.fb_updates = fb_updates
        args.fb_type = fb_type
        args.with_text_embedding = with_t

        # run the test
        test(args)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    test(args)
