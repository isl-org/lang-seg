from collections import defaultdict
import re
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch.nn.functional as F

from modules.models.lseg_vit import forward_vit
from modules.repri_classifier import Classifier, batch_intersectionAndUnionGPU, to_one_hot
from .lsegmentation_module import LSegmentationModule
from .models.lseg_net import BaseModel, LSeg, LSegNet
from encoding.models.sseg.base import up_kwargs
from tqdm import tqdm
# from visdom_logger import VisdomLogger
from torchfusion.utils import VisdomLogger


import random
import time


import os
import clip
import numpy as np

from scipy import signal
import glob

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class LSegModule(LSegmentationModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super(LSegModule, self).__init__(
            data_path, dataset, batch_size, base_lr, max_epochs, **kwargs
        )

        if dataset == "citys":
            self.base_size = 2048
            self.crop_size = 768
        else:
            self.base_size = 520
            self.crop_size = 480

        use_pretrained = True
        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        print('** Use norm {}, {} as the mean and std **'.format(norm_mean, norm_std))

        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

        self.trainset = self.get_trainset(
            dataset,
            augment=kwargs["augment"],
            base_size=self.base_size,
            crop_size=self.crop_size,
        )
        
        self.valset = self.get_valset(
            dataset,
            augment=kwargs["augment"],
            base_size=self.base_size,
            crop_size=self.crop_size,
        )

        use_batchnorm = (
            (not kwargs["no_batchnorm"]) if "no_batchnorm" in kwargs else True
        )
        # print(kwargs)

        labels = self.get_labels('ade20k')

        self.net = LSegNet(
            labels=labels,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=self.crop_size,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
        )
        # something relates to the patch embedding of the vision transformer
        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        self._up_kwargs = up_kwargs
        self.mean = norm_mean
        self.std = norm_std

        self.criterion = self.get_criterion(**kwargs)

    def get_labels(self, dataset):
        labels = []
        path = 'label_files/{}_objectInfo150.txt'.format(dataset)
        assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
        f = open(path, 'r') 
        lines = f.readlines()      
        for line in lines: 
            # one label for training each image but at inference time, we can do open vocab segmentation
            label = line.strip().split(',')[-1].split(';')[0]
            labels.append(label)
        f.close()
        if dataset in ['ade20k']: # remove the header row for ade20k label file only
            labels = labels[1:]
        return labels


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone network",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of featurs that go from encoder to decoder",
        )

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        parser.add_argument(
            "--finetune_weights", type=str, help="load weights to finetune from"
        )

        parser.add_argument(
            "--no-scaleinv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--no-batchnorm",
            default=False,
            action="store_true",
            help="turn off batchnorm",
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
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        return parser

#%% module that adapt the RePRI to LSeg


class LSegNetRePRI(LSegNet):
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):
        super(LSegNet).__init__(labels, path, scale_factor, crop_size, **kwargs)

class LSegmentationRePRIModule(LSegmentationModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super(LSegmentationModule).__init__()
    
    def episodic_validate_RePRI(self, args):

        # freeze the main body of the model
        self.net.eval()
        # total number of test cases / batch size = number of episode, a episode contains a multiple (query, supports) pairs of random classes
        nb_episodes = int(args.test_num / args.batch_size_val)

        # ========== Metrics initialization  ==========

        # TODO: modify 
        H, W = args.image_size, args.image_size
        c = self.net.module.bottleneck_dim
        h = self.net.module.feature_res[0]
        w = self.net.module.feature_res[1]

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
            runtime = 0
            # for each episode, it contains multiple (query, supports) pairs, each pair requires its own \theta parameters of the classifier 
            # to make inference on the pair corresponding query image.
            for e in tqdm(range(nb_episodes)):
                t0 = time.time()
                # NOTE: args.batch_size_val is equivalent to say number of tasks
                features_s = torch.zeros(args.batch_size_val, args.shot, c, h, w).to(dist.get_rank())
                features_q = torch.zeros(args.batch_size_val, 1, c, h, w).to(dist.get_rank())
                gt_s = 255 * torch.ones(args.batch_size_val, args.shot, args.image_size,
                                        args.image_size).long().to(dist.get_rank())
                gt_q = 255 * torch.ones(args.batch_size_val, 1, args.image_size,
                                        args.image_size).long().to(dist.get_rank())
                n_shots = torch.zeros(args.batch_size_val).to(dist.get_rank())
                classes = []  # All classes considered in the tasks

                # =========== Generate tasks and extract features for each task ===============
                with torch.no_grad():
                    for i in range(args.batch_size_val):
                        # load each pair in the episode one by one
                        try:
                            qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iter_loader.next()
                        except:
                            iter_loader = iter(val_loader)
                            qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iter_loader.next()
                        iter_num += 1

                        # place it to the corresponding gpu/cpu/the ith gpu in the cluster
                        q_label = q_label.to(dist.get_rank(), non_blocking=True)
                        spprt_imgs = spprt_imgs.to(dist.get_rank(), non_blocking=True)
                        s_label = s_label.to(dist.get_rank(), non_blocking=True)
                        qry_img = qry_img.to(dist.get_rank(), non_blocking=True)

                        # get the final feature tensor of the support images and the query image
                        f_s = self.net.extract_features(spprt_imgs.squeeze(0)) #[shots, c, h, w]
                        f_q = self.net.extract_features(qry_img) #[1, c, h, w]

                        shot = f_s.size(0)
                        n_shots[i] = shot
                        features_s[i, :shot] = f_s.detach() # add the feature tensor of the shots to the container for each pair in the batch
                        features_q[i] = f_q.detach() # same for the query but only one shot here
                        
                        # store the corresponding labels
                        gt_s[i, :shot] = s_label
                        gt_q[i, 0] = q_label
                        
                        # add individual class label in a batch to the container, recall item() only work for tensor that contains one element only
                        classes.append([class_.item() for class_ in subcls])
                
                # =========== Normalize features along channel dimension ===============
                if args.norm_feat:
                    features_s = F.normalize(features_s, dim=2)
                    features_q = F.normalize(features_q, dim=2)
                
                # =========== Create a callback is args.visdom_port != -1 ===============
                callback = VisdomLogger(port=args.visdom_port) if use_callback else None

                repri_args = {}
                self.RePRI_inference(repri_args)

            
    def RePRI_inference(self, args):
        """few-shot fine-tuning using RePRI; this method is used in self.episodic_validate_RePRI

        Args:
            args (dict): a dictionary that contains a list of key-values for the function,
            populated from the self.episodic_validate_RePRI
        """
        classifier_args = '' # check classifier's constructor to fill in the values

        features_s = args['feature_s']
        features_q = args['feature_q']
        gt_s = args['gt_s']
        gt_q = args['gt_q']
        classes = args['classes']
        callback = args['callback']
        n_shots = args['n_shots']

        # log metrics
        cls_intersection = args['cls_intersection']
        cls_union = args['cls_union']
        IoU = args['IoU']
        iter_num = args['iter_num']
        loss_meter = args['loss_meter']
        H,W  = args['shape']

        # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
        classifier = Classifier(classifier_args)
        classifier.init_prototypes(features_s, features_q, gt_s, gt_q, classes, callback)
        batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
        # deltas_init[run, e, :] = batch_deltas.cpu()

        # =========== Perform RePRI inference ===============

        # train the one layer classifier to learn the optimal average prototype in the support images
        batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback)
        # deltas_final[run, e, :] = batch_deltas
        # t1 = time.time()
        # runtime += t1 - t0

        # perform actual inference on the query image with the prototype learnt from the support shots
        logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w]
        logits = F.interpolate(logits,
                                size=(H, W),
                                mode='bilinear',
                                align_corners=True) # upsample the logit score to the original image size
        probas = classifier.get_probas(logits).detach() # get the probabilty at each spatial location
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

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])
            print('Test: [{}/{}] '
                    'mIoU {:.4f} '
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
                                                                                args.test_num,
                                                                                mIoU,
                                                                                loss_meter=loss_meter,
                                                                                ))

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

class LSegRePRIModule(LSegmentationRePRIModule):
    """This is the end point to be called in the test.py"""

    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super(LSegmentationRePRIModule, self).__init__(
            data_path, dataset, batch_size, base_lr, max_epochs, **kwargs
        )

        #  TODO: to be modified
        self.base_size = 520
        self.crop_size = 480
        use_pretrained = True
        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        print('** Use norm {}, {} as the mean and std **'.format(norm_mean, norm_std))

        # used in train and val dataset loader start
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

        self.trainset = self.get_trainset(
            dataset,
            augment=kwargs["augment"],
            base_size=self.base_size,
            crop_size=self.crop_size,
        )
        
        self.valset = self.get_valset(
            dataset,
            augment=kwargs["augment"],
            base_size=self.base_size,
            crop_size=self.crop_size,
        )
        # used in train and val dataset loader end

        use_batchnorm = (
            (not kwargs["no_batchnorm"]) if "no_batchnorm" in kwargs else True
        )
        # print(kwargs)

        # overrides the parent property
        labels = self.get_labels('fewshot_pascal')

        self.net = LSegNetRePRI(
            labels=labels,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=self.crop_size,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
        )
        # something relates to the patch embedding of the vision transformer
        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

    def get_labels(self, dataset):
        """get the label for the datasset PascalVOC2012

        Args:
            dataset (string): should be 'fewshot_pascal'

        Returns:
            string: a list of class labels in order
        """
        labels = []
        path = 'label_files/{}.txt'.format(dataset)
        assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
        f = open(path, 'r') 
        lines = f.readlines()      
        for line in lines: 
            # one label for each line
            label = line.strip()
            labels.append(label)
        f.close()
        return labels

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            worker_init_fn=lambda x: random.seed(time.time() + x),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone network",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of featurs that go from encoder to decoder",
        )

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        parser.add_argument(
            "--finetune_weights", type=str, help="load weights to finetune from"
        )

        parser.add_argument(
            "--no-scaleinv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--no-batchnorm",
            default=False,
            action="store_true",
            help="turn off batchnorm",
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
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        return parser

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
