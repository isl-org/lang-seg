import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks_zs import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd

import os

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out

class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x

# tanh relu
class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clipRN50x4_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
            "clipRN50x16_vitb32_384": [2, 5, 8, 11],
            "clipRN50x4_vitb32_384": [2, 5, 8, 11],
            "clip_resnet101": [0, 1, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            self.use_pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # self.scratch.output_conv = head

        self.auxlayer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        # cosine similarity as logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        if backbone in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif backbone in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]

        self.scratch.output_conv = head

        self.texts = []
        # original
        label = ['others', '']
        for class_i in range(len(self.label_list)):
            label[1] = self.label_list[class_i]
            text = clip.tokenize(label)
            self.texts.append(text)

    def forward(self, x, class_info):
        texts = [self.texts[class_i] for class_i in class_info]
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        self.logit_scale = self.logit_scale.to(x.device)
        text_features = [self.clip_pretrained.encode_text(text.to(x.device)) for text in texts]

        image_features = self.scratch.head1(path_1)


        imshape = image_features.shape
        image_features = [image_features[i].unsqueeze(0).permute(0,2,3,1).reshape(-1, self.out_c) for i in range(len(image_features))]

        # normalized features
        image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
        text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
        
        logits_per_images = [self.logit_scale * image_feature.half() @ text_feature.t() for image_feature, text_feature in zip(image_features, text_features)]
        outs = [logits_per_image.float().view(1, imshape[2], imshape[3], -1).permute(0,3,1,2) for logits_per_image in logits_per_images]
        out = torch.cat([out for out in outs], dim=0)

        out = self.scratch.output_conv(out)
            
        return out


class LSegNetZS(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, label_list, path=None, scale_factor=0.5, aux=False, use_relabeled=False, use_pretrained=True, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.scale_factor = scale_factor
        self.aux = aux
        self.use_relabeled = use_relabeled
        self.label_list = label_list
        self.use_pretrained = use_pretrained

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


class LSegRN(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_resnet101",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSegRN, self).__init__()

        self.channels_last = channels_last

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            self.use_pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # self.scratch.output_conv = head

        self.auxlayer = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        # cosine similarity as logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

        if backbone in ["clipRN50x16_vitl16_384", "clipRN50x16_vitb32_384"]:
            self.out_c = 768
        elif backbone in ["clipRN50x4_vitl16_384", "clipRN50x4_vitb32_384"]:
            self.out_c = 640
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]

        self.scratch.output_conv = head

        self.texts = []
        # original
        label = ['others', '']
        for class_i in range(len(self.label_list)):
            label[1] = self.label_list[class_i]
            text = clip.tokenize(label)
            self.texts.append(text)

    def forward(self, x, class_info):
        texts = [self.texts[class_i] for class_i in class_info]
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        self.logit_scale = self.logit_scale.to(x.device)
        text_features = [self.clip_pretrained.encode_text(text.to(x.device)) for text in texts]

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = [image_features[i].unsqueeze(0).permute(0,2,3,1).reshape(-1, self.out_c) for i in range(len(image_features))]

        # normalized features
        image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
        text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
        
        logits_per_images = [self.logit_scale * image_feature.half() @ text_feature.t() for image_feature, text_feature in zip(image_features, text_features)]
        outs = [logits_per_image.float().view(1, imshape[2], imshape[3], -1).permute(0,3,1,2) for logits_per_image in logits_per_images]
        out = torch.cat([out for out in outs], dim=0)

        out = self.scratch.output_conv(out)
            
        return out


class LSegRNNetZS(LSegRN):
    """Network for semantic segmentation."""
    def __init__(self, label_list, path=None, scale_factor=0.5, aux=False, use_relabeled=False, use_pretrained=True, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.scale_factor = scale_factor
        self.aux = aux
        self.use_relabeled = use_relabeled
        self.label_list = label_list
        self.use_pretrained = use_pretrained

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

