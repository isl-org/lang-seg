import os
import pathlib

from glob import glob

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
import numpy as np
#import cv2
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

    hparams.callbacks = make_checkpoint_callbacks(hparams.exp_name, hparams.version)

    hparams.logger = pl.loggers.TensorBoardLogger("logs/")

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