import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from .lsegmentation_module_zs import LSegmentationModuleZS
from .models.lseg_net_zs import LSegNetZS, LSegRNNetZS
from encoding.models.sseg.base import up_kwargs
import os
import clip
import numpy as np
from scipy import signal
import glob
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


class LSegModuleZS(LSegmentationModuleZS):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super(LSegModuleZS, self).__init__(
            data_path, dataset, batch_size, base_lr, max_epochs, **kwargs
        )
        label_list = self.get_labels(dataset)
        self.len_dataloader = len(label_list)

        # print(kwargs)
        if kwargs["use_pretrained"] in ['False', False]:
            use_pretrained = False
        elif kwargs["use_pretrained"] in ['True', True]:
            use_pretrained = True

        if kwargs["backbone"] in ["clip_resnet101"]:
            self.net = LSegRNNetZS(
                label_list=label_list,
                backbone=kwargs["backbone"],
                features=kwargs["num_features"],
                aux=kwargs["aux"],
                use_pretrained=use_pretrained,
                arch_option=kwargs["arch_option"],
                block_depth=kwargs["block_depth"],
                activation=kwargs["activation"],
            )
        else:
            self.net = LSegNetZS(
                label_list=label_list,
                backbone=kwargs["backbone"],
                features=kwargs["num_features"],
                aux=kwargs["aux"],
                use_pretrained=use_pretrained,
                arch_option=kwargs["arch_option"],
                block_depth=kwargs["block_depth"],
                activation=kwargs["activation"],
            )

    def get_labels(self, dataset):
        labels = []
        path = 'label_files/fewshot_{}.txt'.format(dataset)
        assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
        f = open(path, 'r') 
        lines = f.readlines()      
        for line in lines: 
            label = line.strip()
            labels.append(label)
        f.close()
        print(labels)
        return labels

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModuleZS.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="vitb16_384",
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

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['relu', 'lrelu', 'tanh'],
            default="relu",
            help="use which activation to activate the block",
        )

        return parser
