from collections import defaultdict
import itertools
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from modules.lseg_module_zs import LSegModuleZS
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.data.dataset import FSSDataset
from repri_classifier import Classifier, batch_intersectionAndUnionGPU, to_one_hot
from types import SimpleNamespace
import pandas as pd

LAB_COMPUTER_ENV = True

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
            default=1
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
            default='/Users/maxxyouu/Desktop/lang-seg-fork/data/' if not LAB_COMPUTER_ENV else 'E:\jose_tasks\lang-seg\data'
            # for home use the directory: '/Users/maxxyouu/Desktop/lang-seg-fork/data/'
            # for lab computer: 'E:\jose_tasks\lang-seg\data'
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
    c = 512
    h = 240
    w = 240

    # intialize the container to store the results
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

            # =========== Generate tasks and extract features for each task - a pair of query and supports ==============
            with torch.no_grad():
                for i in range(args.batch_size_val):
                    # load each pair in the episode one by one
                    try:
                        batch = next(iter_loader)
                    except:
                        # for multiple runs
                        iter_loader = iter(args.val_loader)
                        batch = next(iter_loader)

                    qry_img = batch['query_img']
                    q_label = batch['query_mask']
                    spprt_imgs = batch['support_imgs']
                    s_label = batch['support_masks']
                    subcls = batch['class_id']
                    iter_num += 1

                    # place it to the corresponding gpu/cpu/the ith gpu in the cluster
                    q_label = q_label.to(device)
                    if spprt_imgs == []:
                        spprt_imgs = None # no support images
                        s_label = None
                    else:
                        spprt_imgs = spprt_imgs.to(device)
                        s_label = s_label.to(device)
                    qry_img = qry_img.to(device)

                    # get the final feature tensor of the support images and the query image
                    f_s, t_s, _ = model.extract_features(spprt_imgs.squeeze(0) if spprt_imgs is not None else None, subcls)
                    f_q, _, _ = model.extract_features(qry_img, subcls)
                    t_s = t_s[-1]

                    if spprt_imgs is not None:
                        shot = f_s.size(0)
                        n_shots[i] = shot
                        features_s[i, :shot] = f_s.detach() # add the feature tensor of the shots to the container for each pair in the batch
                        gt_s[i, :shot] = s_label
                        
                    features_q[i] = f_q.detach() # same for the query but only one shot here
                    text_s[i] = t_s.detach()                    
                    gt_q[i, 0] = q_label
                    
                    # add individual class label in a batch to the container, recall item() only work for tensor that contains one element only
                    classes.append([class_.item() for class_ in subcls])
            
            # =========== Normalize features along channel dimension ===============
            if args.shot == 0:
                features_s = None
            else:
                features_s = F.normalize(features_s, dim=2)
            features_q = F.normalize(features_q, dim=2)
            text_s = F.normalize(text_s, dim=2)

            # =========== Create a callback is args.visdom_port != -1 ===============
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

    return val_IoUs.mean(), val_losses.mean()

def load_checkpoint(module_def, args):
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
    module = load_checkpoint(module_def, args)

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
    # 2. try different combination of adapt_iter, lr, and param updates (the frequency);  DONE
    # 3. try the same set of hyperparameters with oracle DONE
    # 4. try with and without the shannon entropy and/or KL divergence
    params = {
        'module': module,
        'image_size':  image_size,
        'test_num': len(dataset.img_metadata), # total number of test cases
        'batch_size_val': 2, # NOTE: this is different than the args.bsz
        'n_runs': args.n_run, # repeat the experiment 1 time
        'shot': args.nshot,
        'val_loader': dataloader,
        'benchmark': args.dataset, # pascal
        # the following are for the RePRI classifier
        'temperature': args.temp,
        'adapt_iter': args.adapt_iter,
        'weights': [1.0, 'auto', 'auto'],
        'cls_lr': args.cls_lr,
        'FB_param_update': args.fb_updates,
        'cls_visdom_freq': 5, # might not need it
        # NOTE: we only need the following when it is a oracle experiment
        'FB_param_type': args.fb_type, 
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
    mIoU, losses = episodic_validate(SimpleNamespace(**params))
    return mIoU


def hyperparameter_tuning():
    # folds = [0]
    # weights_paths = ['./checkpoints/pascal_fold0.ckpt']
    # with_text_embeddings = [True, False]

    # hyperparameter space
    RUNS = 1 # this is the same across all experiments
    shots = [0, 1] # number shot outside of this will cause CUDA out of memory
    temperatures = [20]
    iterations = [50]
    learning_rates = [0.025]
    fb_params = [[10, 30]]
    fb_params_types = ['oracle'] # 'oracle'
    folds = [0, 1, 2, 3]
    weights_paths = ['./checkpoints/pascal_fold0.ckpt', './checkpoints/pascal_fold1.ckpt', './checkpoints/pascal_fold2.ckpt', './checkpoints/pascal_fold3.ckpt']

    for fold, weight_path in zip(folds, weights_paths):

        shots_col_data = []
        tmp_col_data = []
        lr_col_data = []
        fb_params_data = []
        fb_type_data = []
        fold_data = []
        miou_data = []
        col_names = ['n_shots', 'temperature', 'learning_rate', 'fb_params', 'fb_type', 'fold', 'mIoU']

        hy_params = itertools.product(shots, temperatures, iterations, learning_rates, fb_params, fb_params_types)
        for shot, tmp, iter, lr, fb_updates, fb_type in hy_params:
            print('nshot-{}; temperature-{}; lr-{}; fb-{}; fb_type-{}; fold-{}'.format(shot, tmp, iter, lr, fb_updates, fb_type, fold))
            args = Options().parse()
            torch.manual_seed(args.seed)

            # classifier hyper parameter changing
            args.n_run = RUNS
            args.nshot = shot
            args.temp = tmp
            args.adapt_iter = iter
            args.cls_lr = lr
            args.fb_updates = fb_updates
            args.fb_type = fb_type
            # args.with_text_embedding = with_t
            args.fold = fold
            args.weights = weight_path

            # run the test
            miou = test(args)

            # store column data for excel
            shots_col_data.append(shot)
            tmp_col_data.append(tmp)
            lr_col_data.append(lr)
            fb_params_data.append(fb_updates)
            fb_type_data.append('non-oracle' if 'oracle' not in fb_type else 'oracle')
            fold_data.append(fold)
            miou_data.append(miou)

        # Create DataFrame from multiple lists
        df = pd.DataFrame(list(zip(shots_col_data, tmp_col_data, lr_col_data, fb_params_data, fb_type_data, fold_data, miou_data)), columns=col_names)
        df.to_excel('./fold-{}_nRun-{}_fbType-{}_comment-{}.xlsx'.format(fold, args.n_run, 'nonOracle' if 'oracle' not in fb_params_types[0] else 'oracle', ''))
        print('===============================Finish writing experiment data for fold {}===================================='.format(fold))


if __name__ == "__main__":
    # args = Options().parse()
    # torch.manual_seed(args.seed)
    # args.temp = 100
    # args.adapt_iter = 100
    # args.fb_updates = [i for i in range(0, args.adapt_iter)]
    # args.fb_type = 'oracle'
    # args.cls_lr = 0.025
    # args.n_run = 1
    # test(args)

    hyperparameter_tuning()