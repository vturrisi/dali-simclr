import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from models.barlow_twins import BarlowTwins
from models.dali import DaliBarlowTwins, DaliSimCLR, DaliSimSiam, DaliBYOL, DaliNNCLR
from models.simclr import SimCLR
from models.simsiam import SimSiam
from models.byol import BYOL
from models.nnclr import NNCLR

from utils.classification_dataloader import prepare_data as prepare_data_classification
from utils.contrastive_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_multicrop_transform,
    prepare_transform,
)
from utils.epoch_checkpointer import EpochCheckpointer


def parse_args():
    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "stl10",
        "imagenet",
        "imagenet100",
    ]

    SUPPORTED_NETWORKS = ["resnet18", "resnet50"]

    SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

    SUPPORTED_SCHEDULERS = [
        "reduce",
        "cosine",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS, type=str)
    parser.add_argument("encoder", choices=SUPPORTED_NETWORKS, type=str)

    parser.add_argument(
        "--method", choices=["simclr", "barlow_twins", "simsiam", "byol", "nnclr"], default=None
    )

    # optimizer
    parser.add_argument("--optimizer", default="sgd", choices=SUPPORTED_OPTIMIZERS, type=str)
    parser.add_argument("--lars", action="store_true")

    # scheduler
    parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
    parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
    parser.add_argument("--no_lr_scheduler_for_pred_head", action="store_true")

    # general settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--classifier_lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--zero_init_residual", action="store_true")

    # projector
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # extra training settings
    parser.add_argument("--resume_training_from", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--precision", type=int, default=16)

    # dataset path
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)

    # extra dataloader settings
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--n_crops", type=int, default=2)
    parser.add_argument("--n_small_crops", type=int, default=6)
    parser.add_argument("--brightness", type=float, default=0.8)
    parser.add_argument("--contrast", type=float, default=0.8)
    parser.add_argument("--saturation", type=float, default=0.8)
    parser.add_argument("--hue", type=float, default=0.2)
    parser.add_argument("--gaussian_prob", type=float, default=0.5)
    parser.add_argument("--solarization_prob", type=float, default=0)
    # this only works for imagenet
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--dali_device", type=str, default="gpu")
    parser.add_argument("--last_batch_fill", action="store_true")

    # extra simclr settings
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--supervised", action="store_true")

    # extra barlow twins settings
    parser.add_argument("--lamb", type=float, default=5e-3)
    parser.add_argument("--scale_loss", type=float, default=0.025)
    parser.add_argument("--asymmetric_augmentations", action="store_true")

    # extra simsiam settings
    parser.add_argument("--pred_hidden_dim", type=int, default=512)

    # extra byol settings
    parser.add_argument("--base_tau_momentum", default=0.99, type=float)
    parser.add_argument("--final_tau_momentum", default=1.0, type=float)

    # extra nnclr settings
    parser.add_argument("--queue_size", type=int, default=98304)

    # multi-head stuff
    parser.add_argument("--n_heads", type=int, default=2)

    # wandb
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--offline", action="store_true")

    args = parser.parse_args()

    args.transform_kwargs = {}
    if args.dataset == "cifar10":
        args.n_classes = 10
    elif args.dataset == "cifar100":
        args.n_classes = 100
    elif args.dataset == "stl10":
        args.n_classes = 10
    else:
        if args.dataset == "imagenet":
            args.n_classes = 1000
        else:
            args.n_classes = 100

        if args.asymmetric_augmentations:
            args.transform_kwargs = [
                dict(
                    brightness=args.brightness,
                    contrast=args.contrast,
                    saturation=args.saturation,
                    hue=args.hue,
                    gaussian_prob=1.0,
                    solarization_prob=0.0,
                ),
                dict(
                    brightness=args.brightness,
                    contrast=args.contrast,
                    saturation=args.saturation,
                    hue=args.hue,
                    gaussian_prob=0.1,
                    solarization_prob=0.2,
                ),
            ]
        else:
            args.transform_kwargs = dict(
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                hue=args.hue,
                gaussian_prob=args.gaussian_prob,
                solarization_prob=args.solarization_prob,
            )

    if args.asymmetric_augmentations:
        assert args.dataset in ["imagenet", "imagenet100"]

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    # adjust lr according to batch size
    args.lr = args.lr * args.batch_size * len(args.gpus) / 256

    assert args.method is not None

    return args


def main():
    seed_everything(5)

    args = parse_args()

    if args.method == "simclr":
        if args.dali:
            model = DaliSimCLR(args)
        else:
            model = SimCLR(args)
    elif args.method == "barlow_twins":
        if args.dali:
            model = DaliBarlowTwins(args)
        else:
            model = BarlowTwins(args)
    elif args.method == "simsiam":
        if args.dali:
            model = DaliSimSiam(args)
        else:
            model = SimSiam(args)
    elif args.method == "byol":
        if args.dali:
            model = DaliBYOL(args)
        else:
            model = BYOL(args)
    elif args.method == "nnclr":
        if args.dali:
            model = DaliNNCLR(args)
        else:
            model = NNCLR(args)

    # contrastive dataloader
    if not args.dali:
        # asymmetric augmentations on barlow twins
        if args.asymmetric_augmentations:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.multicrop:
            assert not args.asymmetric_augmentations

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform, size_crops=size_crops, n_crops=[args.n_crops, args.n_small_crops]
            )
        else:
            transform = prepare_n_crop_transform(transform, n_crops=2)

        train_dataset = prepare_datasets(
            args.dataset,
            data_folder=args.data_folder,
            train_dir=args.train_dir,
            transform=transform,
        )
        train_loader = prepare_dataloaders(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # normal dataloader
    _, val_loader = prepare_data_classification(
        args.dataset,
        data_folder=args.data_folder,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)
        # lr logging
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # epoch checkpointer
    callbacks.append(EpochCheckpointer(args, frequency=25))

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=[*args.gpus],
        logger=wandb_logger if args.wandb else None,
        distributed_backend="ddp",
        precision=args.precision,
        sync_batchnorm=True,
        resume_from_checkpoint=args.resume_training_from,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
