import os
import pathlib

from glob import glob

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
import numpy as np
import cv2
import random
import math
from torchvision import transforms


def do_training(hparams, model_constructor):
    # instantiate model
    model = model_constructor(**vars(hparams))
    # set all sorts of training parameters
    hparams.gpus = -1
    hparams.accelerator = "ddp"
    hparams.benchmark = True

    if hparams.dry_run:
        print("Doing a dry run")
        hparams.overfit_batches = hparams.batch_size

    if not hparams.no_resume:
        hparams = set_resume_parameters(hparams)

    if not hasattr(hparams, "version") or hparams.version is None:
        hparams.version = 0

    hparams.sync_batchnorm = True

    ttlogger = pl.loggers.TestTubeLogger(
        "checkpoints", name=hparams.exp_name, version=hparams.version
    )

    hparams.callbacks = make_checkpoint_callbacks(hparams.exp_name, hparams.version)

    wblogger = get_wandb_logger(hparams)
    hparams.logger = [wblogger, ttlogger]

    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    

def get_default_argument_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )

    parser.add_argument(
        "--exp_name", type=str, required=True, help="name your experiment"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="run on batch of train/val/test",
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        default=False,
        help="resume if we have a checkpoint",
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="accumulate N batches for gradient computation",
    )

    parser.add_argument(
        "--max_epochs", type=int, default=200, help="maximum number of epochs"
    )

    parser.add_argument(
        "--project_name", type=str, default="lightseg", help="project name for logging"
    )

    return parser


def make_checkpoint_callbacks(exp_name, version, base_path="checkpoints", frequency=1):
    version = 0 if version is None else version

    base_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{base_path}/{exp_name}/version_{version}/checkpoints/",
        save_last=True,
        verbose=True,
    )

    val_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc_epoch",
        dirpath=f"{base_path}/{exp_name}/version_{version}/checkpoints/",
        filename="result-{epoch}-{val_acc_epoch:.2f}",
        mode="max",
        save_top_k=3,
        verbose=True,
    )

    return [base_callback, val_callback]


def get_latest_version(folder):
    versions = [
        int(pathlib.PurePath(path).name.split("_")[-1])
        for path in glob(f"{folder}/version_*/")
    ]

    if len(versions) == 0:
        return None

    versions.sort()
    return versions[-1]


def get_latest_checkpoint(exp_name, version):
    while version > -1:
        folder = f"./checkpoints/{exp_name}/version_{version}/checkpoints/"

        latest = f"{folder}/last.ckpt"
        if os.path.exists(latest):
            return latest, version

        chkpts = glob(f"{folder}/epoch=*.ckpt")

        if len(chkpts) > 0:
            break

        version -= 1

    if len(chkpts) == 0:
        return None, None

    latest = max(chkpts, key=os.path.getctime)

    return latest, version


def set_resume_parameters(hparams):
    version = get_latest_version(f"./checkpoints/{hparams.exp_name}")

    if version is not None:
        latest, version = get_latest_checkpoint(hparams.exp_name, version)
        print(f"Resuming checkpoint {latest}, exp_version={version}")

        hparams.resume_from_checkpoint = latest
        hparams.version = version

        wandb_file = "checkpoints/{hparams.exp_name}/version_{version}/wandb_id"
        if os.path.exists(wandb_file):
            with open(wandb_file, "r") as f:
                hparams.wandb_id = f.read()
    else:
        version = 0

    return hparams


def get_wandb_logger(hparams):
    exp_dir = f"checkpoints/{hparams.exp_name}/version_{hparams.version}/"
    id_file = f"{exp_dir}/wandb_id"

    if os.path.exists(id_file):
        with open(id_file) as f:
            hparams.wandb_id = f.read()
    else:
        hparams.wandb_id = None

    logger = pl.loggers.WandbLogger(
        save_dir="checkpoints",
        project=hparams.project_name,
        name=hparams.exp_name,
        id=hparams.wandb_id,
    )

    if hparams.wandb_id is None:
        _ = logger.experiment

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

        with open(id_file, "w") as f:
            f.write(logger.version)

    return logger


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        letter_box=False,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        self.__letter_box = letter_box

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def make_letter_box(self, sample):
        top = bottom = (self.__height - sample.shape[0]) // 2
        left = right = (self.__width - sample.shape[1]) // 2
        sample = cv2.copyMakeBorder(
            sample, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0
        )
        return sample

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__letter_box:
            sample["image"] = self.make_letter_box(sample["image"])

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                if self.__letter_box:
                    sample["disparity"] = self.make_letter_box(sample["disparity"])

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

                if self.__letter_box:
                    sample["depth"] = self.make_letter_box(sample["depth"])

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            if self.__letter_box:
                sample["mask"] = self.make_letter_box(sample["mask"])

            sample["mask"] = sample["mask"].astype(bool)

        return sample
