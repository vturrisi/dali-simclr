import os
import sys
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.lars import LARSWrapper
from utils.metrics import accuracy_at_k, weighted_mean


def static_lr(get_lr, param_group_indexes, lrs_to_replace):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

    def configure_optimizers(self):
        args = self.args

        # select optimizer
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{args.optimizer} not in (sgd, adam)")

        if hasattr(self, "classifier"):
            classifier_parameters = self.classifier.parameters()

            if args.no_lr_scheduler_for_pred_head:
                predictor_parameters = self.predictor.parameters()
                other_parameters = (
                    p
                    for name, p in self.named_parameters()
                    if not any(s in name for s in ["classifier", "predictor", "momentum"])
                )
                parameters = [
                    {"params": other_parameters},
                    {"params": classifier_parameters, "lr": args.classifier_lr, "weight_decay": 0},
                    {"params": predictor_parameters},
                ]
            else:
                other_parameters = (
                    p
                    for name, p in self.named_parameters()
                    if not any(s in name for s in ["classifier", "momentum"])
                )
                parameters = [
                    {"params": other_parameters},
                    {"params": classifier_parameters, "lr": args.classifier_lr, "weight_decay": 0},
                ]
        else:
            if args.no_lr_scheduler_for_pred_head:
                predictor_parameters = self.predictor.parameters()
                other_parameters = (
                    p
                    for name, p in self.named_parameters()
                    if not any(s in name for s in ["predictor", "momentum"])
                )
                parameters = [
                    {"params": other_parameters},
                    {"params": predictor_parameters},
                ]
            else:
                parameters = [
                    {
                        "params": (
                            p
                            for name, p in self.named_parameters()
                            if not any(s in name for s in ["momentum"])
                        )
                    },
                ]

        optimizer = optimizer(
            parameters, lr=args.lr, weight_decay=args.weight_decay, **args.extra_optimizer_args,
        )
        if args.lars:
            optimizer = LARSWrapper(optimizer)

        if args.scheduler == "none":
            return optimizer
        else:
            assert args.scheduler in ["warmup_cosine", "cosine", "step"]

            if args.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=10, max_epochs=args.epochs, warmup_start_lr=0.003,
                )
            elif args.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, args.epochs)
            else:
                scheduler = MultiStepLR(optimizer, args.lr_decay_steps)

            if args.no_lr_scheduler_for_pred_head:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=(2,),
                    lrs_to_replace=(args.lr,),
                )
                scheduler.get_lr = partial_fn
            return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        X, target = batch
        batch_size = X.size(0)

        output = self(X)
        loss = F.cross_entropy(output, target)

        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = weighted_mean(outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outputs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, prog_bar=True, sync_dist=True)


class Model(BaseModel):
    """
    Implementation of the base model that automatically
    creates a linear classifier for online evaluation.
    The linear classifier is automatically detached from the computational graph.
    """

    def __init__(self, args):
        super().__init__(args)

        assert args.encoder in ["resnet18", "resnet50"]
        from torchvision.models import resnet18, resnet50

        self.base_model = {"resnet18": resnet18, "resnet50": resnet50}[args.encoder]

        # initialize encoder
        self.encoder = self.base_model(zero_init_residual=args.zero_init_residual)
        self.features_size = self.encoder.inplanes
        # remove fc layer
        self.encoder.fc = nn.Identity()
        if args.cifar:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.encoder.maxpool = nn.Identity()

        self.classifier = nn.Linear(self.features_size, args.n_classes)

    def forward(self, X, classify_only=True):
        feat = self.encoder(X)
        # stop gradients from the classifier
        y = self.classifier(feat.detach())

        if classify_only:
            return y

        return feat, y
