import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.lseg_module_zs import LSegModuleZS
from additional_utils.models import LSeg_MultiEvalModule
from fewshot_data.common.logger import Logger, AverageMeter
from fewshot_data.common.vis import Visualizer
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.common import utils
from fewshot_data.data.dataset import FSSDataset


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
            default="resnet50",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ade20k",
            help="dataset name (default: pascal12)",
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
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default=None, help="checkpoint to test"
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
            default=1
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
            default='fewshot_data/Datasets_HSN'
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


def test(args):
    module_def = LSegModuleZS

    module = module_def.load_from_checkpoint(
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

    Evaluator.initialize()
    if args.backbone in ["clip_resnet101"]:
        FSSDataset.initialize(img_size=480, datapath=args.datapath, use_original_imgsize=False, imagenet_norm=True)
    else:
        FSSDataset.initialize(img_size=480, datapath=args.datapath, use_original_imgsize=False)
    # dataloader
    args.benchmark = args.dataset
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    model = module.net.eval().cuda()
    # model = module.net.model.cpu()

    print(model)

    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )  

    f = open("logs/fewshot/log_fewshot-test_nshot{}_{}.txt".format(args.nshot, args.dataset), "a+")

    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        image = batch['query_img']
        target = batch['query_mask']
        class_info = batch['class_id']
        # pred_mask = evaluator.parallel_forward(image, class_info)
        pred_mask = model(image, class_info)
        # assert pred_mask.argmax(dim=1).size() == batch['query_mask'].size()
        # 2. Evaluate prediction
        if args.benchmark == 'pascal' and batch['query_ignore_idx'] is not None:
            query_ignore_idx = batch['query_ignore_idx']
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.argmax(dim=1), target, query_ignore_idx)
        else:
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.argmax(dim=1), target)

        average_meter.update(area_inter, area_union, class_info, loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    test_miou, test_fb_iou = average_meter.compute_iou()

    Logger.info('Fold %d, %d-shot ==> mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, args.nshot, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
    f.write('{}\n'.format(args.weights))
    f.write('Fold %d, %d-shot ==> mIoU: %5.2f \t FB-IoU: %5.2f\n' % (args.fold, args.nshot, test_miou.item(), test_fb_iou.item()))
    f.close()
                


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    test(args)
