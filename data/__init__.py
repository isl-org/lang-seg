import copy

import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import encoding.datasets as enc_ds

encoding_datasets = {
    x: functools.partial(enc_ds.get_dataset, x)
    for x in ["coco", "ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
}


def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys()) 
