import types
import time
import random
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from argparse import ArgumentParser

import pytorch_lightning as pl

from encoding.models import get_segmentation_model
from encoding.nn import SegmentationLosses

from encoding.utils import batch_pix_accuracy, batch_intersection_union

# add mixed precision
import torch.cuda.amp as amp
import numpy as np
from encoding.utils.metrics import SegmentationMetric

# get fewshot dataloader
from fewshot_data.model.hsnet import HypercorrSqueezeNetwork
from fewshot_data.common.logger import Logger, AverageMeter
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.common import utils
from fewshot_data.data.dataset import FSSDataset
        
class Fewshot_args:
    datapath = 'fewshot_data/Datasets_HSN'
    benchmark = 'pascal'
    logpath = ''
    nworker = 8
    bsz = 20
    fold = 0
    

class LSegmentationModuleZS(pl.LightningModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super().__init__()

        self.batch_size = batch_size 
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr

        self.epochs = max_epochs
        self.other_kwargs = kwargs
        self.enabled = False #True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)
        # for whether fix the encoder or not
        self.fixed_encoder = True if kwargs["use_pretrained"] in ['clip_fixed'] else False

        # fewshot hyperparameters
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.args = self.get_fewshot_args()
        if data_path:
            self.args.datapath = data_path
        self.args.logpath = self.other_kwargs["logpath"]
        self.args.benchmark = dataset
        self.args.bsz = self.batch_size
        self.args.fold = self.other_kwargs["fold"]
        self.args.nshot =  self.other_kwargs["nshot"]
        self.args.finetune_mode = self.other_kwargs["finetune_mode"]
        Logger.initialize(self.args, training=True)
        Evaluator.initialize()
        if kwargs["backbone"] in ["clip_resnet101"]:
            FSSDataset.initialize(img_size=480, datapath=self.args.datapath, use_original_imgsize=False, imagenet_norm=True)
        else:
            FSSDataset.initialize(img_size=480, datapath=self.args.datapath, use_original_imgsize=False)
        self.best_val_miou = float('-inf')
        self.num_classes = 2
        self.labels = ['others', '']

        self.fewshot_trn_loss = 100
        self.fewshot_trn_miou = 0
        self.fewshot_trn_fb_iou = 0

    def get_fewshot_args(self):
        return Fewshot_args()
        
    def forward(self, x, class_info):
        return self.net(x, class_info)
    

    def training_step(self, batch, batch_nb):
        if self.args.finetune_mode:
            if self.args.nshot == 5:
                bshape = batch['support_imgs'].shape
                img = batch['support_imgs'].view(-1, bshape[2], bshape[3], bshape[4])
                target = batch['support_masks'].view(-1, bshape[3], bshape[4])
                class_info = batch['class_id']
                for i in range(1, 5):
                    class_info = torch.cat([class_info, batch['class_id']])
                with amp.autocast(enabled=self.enabled):
                    out = self(img, class_info)
                    loss = self.criterion(out, target)
                    loss = self.scaler.scale(loss)
                self.log("train_loss", loss)
                # 3. Evaluate prediction
                if self.args.benchmark == 'pascal' and batch['support_ignore_idxs'] is not None:
                    query_ignore_idx = batch['support_ignore_idxs'].view(-1, bshape[3], bshape[4])
                    area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target, query_ignore_idx)
                else:
                    area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target)
            else:
                img = batch['support_imgs'].squeeze(1)
                target = batch['support_masks'].squeeze(1)
                class_info = batch['class_id']
                with amp.autocast(enabled=self.enabled):
                    out = self(img, class_info)
                    loss = self.criterion(out, target)
                    loss = self.scaler.scale(loss)
                self.log("train_loss", loss)
                # 3. Evaluate prediction
                if self.args.benchmark == 'pascal' and batch['support_ignore_idxs'] is not None:
                    query_ignore_idx = batch['support_ignore_idxs'].squeeze(1)
                    area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target, query_ignore_idx)
                else:
                    area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target)
        else:
            img = torch.cat([batch['support_imgs'].squeeze(1), batch['query_img']], dim=0)
            target = torch.cat([batch['support_masks'].squeeze(1), batch['query_mask']], dim=0)
            class_info=torch.cat([batch['class_id'], batch['class_id']], dim=0)
            with amp.autocast(enabled=self.enabled):
                out = self(img, class_info)
                loss = self.criterion(out, target)
                loss = self.scaler.scale(loss)

            self.log("train_loss", loss)
            # 3. Evaluate prediction
            if self.args.benchmark == 'pascal' and batch['query_ignore_idx'] is not None:
                query_ignore_idx = torch.cat([batch['support_ignore_idxs'].squeeze(1), batch['query_ignore_idx']], dim=0)
                area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target, query_ignore_idx)
            else:
                area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target)
        self.train_average_meter.update(area_inter, area_union, class_info, loss.detach().clone())
        if self.global_rank == 0:
            return_value = self.train_average_meter.write_process(batch_nb, self.len_train_dataloader, self.current_epoch, write_batch_idx=50)
            if return_value is not None:
                iou, fb_iou = return_value
                self.log("fewshot_train_iou", iou)
                self.log("fewshot_trainl_fb_iou", fb_iou)

        return loss

    def training_epoch_end(self, outs):
        if self.global_rank == 0:
            self.train_average_meter.write_result('Training', self.current_epoch)
        self.fewshot_trn_loss = utils.mean(self.train_average_meter.loss_buf)
        self.fewshot_trn_miou, self.fewshot_trn_fb_iou = self.train_average_meter.compute_iou()

        self.log("fewshot_trn_loss", self.fewshot_trn_loss)
        self.log("fewshot_trn_miou", self.fewshot_trn_miou)
        self.log("fewshot_trn_fb_iou", self.fewshot_trn_fb_iou)

    def validation_step(self, batch, batch_nb):
        if self.args.finetune_mode and self.args.nshot == 5:
            bshape = batch['query_img'].shape
            img = batch['query_img'].view(-1, bshape[2], bshape[3], bshape[4])
            target = batch['query_mask'].view(-1, bshape[3], bshape[4])
            class_info = batch['class_id']
            for i in range(1, 5):
                class_info = torch.cat([class_info, batch['class_id']])
            out = self(img, class_info)
            val_loss = self.criterion(out, target)
            # 3. Evaluate prediction
            if self.args.benchmark == 'pascal' and batch['query_ignore_idx'] is not None:
                query_ignore_idx = batch['query_ignore_idx'].view(-1, bshape[3], bshape[4])
                area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target, query_ignore_idx)
            else:
                area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target)
        else:
            img = batch['query_img'].squeeze(1)
            target = batch['query_mask'].squeeze(1)
            class_info = batch['class_id']
            out = self(img, class_info)
            val_loss = self.criterion(out, target)
            # 3. Evaluate prediction
            if self.args.benchmark == 'pascal' and batch['query_ignore_idx'] is not None:
                query_ignore_idx = batch['query_ignore_idx'].squeeze(1)
                area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target, query_ignore_idx)
            else:
                area_inter, area_union = Evaluator.classify_prediction(out.argmax(dim=1), target)

        self.val_average_meter.update(area_inter, area_union, class_info, val_loss.detach().clone())
        if self.global_rank == 0:
            return_value = self.val_average_meter.write_process(batch_nb, self.len_val_dataloader, self.current_epoch, write_batch_idx=50) 
            if return_value is not None:
                iou, fb_iou = return_value
                self.log("fewshot_val_iou", iou)
                self.log("fewshot_val_fb_iou", fb_iou)

    
    def validation_epoch_end(self, outs):
        if self.global_rank == 0:
            self.val_average_meter.write_result('Validation', self.current_epoch)
        val_loss = utils.mean(self.val_average_meter.loss_buf)
        val_miou, val_fb_iou = self.val_average_meter.compute_iou()
        self.log("fewshot_val_loss", val_loss)
        self.log("fewshot_val_miou", val_miou)
        self.log("fewshot_val_fb_iou", val_fb_iou)

        if self.global_rank == 0:
            Logger.tbd_writer.add_scalars('fewshot_data/data/loss', {'trn_loss': self.fewshot_trn_loss, 'val_loss': val_loss}, self.current_epoch)
            Logger.tbd_writer.add_scalars('fewshot_data/data/miou', {'trn_miou': self.fewshot_trn_miou, 'val_miou': val_miou}, self.current_epoch)
            Logger.tbd_writer.add_scalars('fewshot_data/data/fb_iou', {'trn_fb_iou': self.fewshot_trn_fb_iou, 'val_fb_iou': val_fb_iou}, self.current_epoch)
            Logger.tbd_writer.flush()
            if self.current_epoch + 1 == self.epochs:
                Logger.tbd_writer.close()
                Logger.info('==================== Finished Training ====================')

        threshold_epoch = 3
        if self.args.benchmark in ['pascal', 'coco'] and self.current_epoch >= threshold_epoch:
            print('End this loop!')
            exit()

    def configure_optimizers(self):
        # if we want to fix the encoder
        if self.fixed_encoder:
            params_list = [
                {"params": self.net.pretrained.model.parameters(), "lr": 0},
            ]
            params_list.append(
                    {"params": self.net.pretrained.act_postprocess1.parameters(), "lr": self.base_lr}
                )
            params_list.append(
                    {"params": self.net.pretrained.act_postprocess2.parameters(), "lr": self.base_lr}
                )
            params_list.append(
                    {"params": self.net.pretrained.act_postprocess3.parameters(), "lr": self.base_lr}
                )
            params_list.append(
                    {"params": self.net.pretrained.act_postprocess4.parameters(), "lr": self.base_lr}
                )
        else:
            params_list = [
                {"params": self.net.pretrained.parameters(), "lr": self.base_lr},
            ]

        if hasattr(self.net, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.net.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "scale_inv_conv"):
            print(self.net.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.net.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.net.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        if self.other_kwargs["midasproto"]:
            print("Using midas optimization protocol")
            
            opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )
        else:
            opt = torch.optim.SGD(
                params_list,
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.other_kwargs["weight_decay"],
            )

            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )
        return [opt], [sch]

    def train_dataloader(self):
        if self.args.finetune_mode:
            dataloader = FSSDataset.build_dataloader(
                self.args.benchmark, 
                self.args.bsz, 
                self.args.nworker, 
                self.args.fold,
                'test', 
                self.args.nshot)
        else:
            dataloader = FSSDataset.build_dataloader(
                self.args.benchmark, 
                self.args.bsz, 
                self.args.nworker, 
                self.args.fold,
                'trn')

        self.len_train_dataloader = len(dataloader) // torch.cuda.device_count()
        self.train_average_meter = AverageMeter(dataloader.dataset)
        return dataloader

    def val_dataloader(self):
        self.val_iou = SegmentationMetric(self.num_classes)
        if self.args.finetune_mode:
            dataloader = FSSDataset.build_dataloader(
                self.args.benchmark, 
                self.args.bsz, 
                self.args.nworker, 
                self.args.fold,
                'test', 
                self.args.nshot)
        else:
            dataloader = FSSDataset.build_dataloader(
                            self.args.benchmark, 
                            self.args.bsz, 
                            self.args.nworker, 
                            self.args.fold,
                            'val')
        self.len_val_dataloader = len(dataloader) // torch.cuda.device_count()
        self.val_average_meter = AverageMeter(dataloader.dataset)
        return dataloader


    def criterion(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", 
            type=str, 
            default='',
            help="path where dataset is stored"
        )
        parser.add_argument(
            "--dataset",
            type=str, 
            default='pascal', 
            choices=['pascal', 'coco', 'fss'],
        )
        parser.add_argument(
            "--batch_size", type=int, default=20, help="size of the batches"
        )
        parser.add_argument(
            "--base_lr", type=float, default=0.004, help="learning rate"
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight_decay"
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--aux-weight",
            type=float,
            default=0.2,
            help="Auxilary loss weight (default: 0.2)",
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
            "--midasproto", action="store_true", default=False, help="midasprotocol"
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            default=False,
            help="Use extended augmentations",
        )
        parser.add_argument(
            "--use_relabeled",
            action="store_true",
            default=False,
            help="Use extended augmentations",
        )

        parser.add_argument(
            "--nworker", 
            type=int, 
            default=8
            )

        parser.add_argument(
            "--fold", 
            type=int, 
            default=0, 
            choices=[0, 1, 2, 3]
            )

        parser.add_argument(
            "--logpath", 
            type=str, 
            default=''
            )
        
        parser.add_argument(
            "--nshot", 
            type=int, 
            default=0 #1
            )
        parser.add_argument(
            "--finetune_mode", 
            action="store_true", 
            default=False, 
            help="whether finetune or not"
        )


        return parser
