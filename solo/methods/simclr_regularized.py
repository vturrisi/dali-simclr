# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.simclr import simclr_loss_func
from solo.losses.manifold_regularizer import ManifoldRegularizer
from solo.utils.weight_schedulers import IntervalScheduler, StepScheduler, TriangleScheduler, WarmupScheduler, ConstantScheduler
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class SimCLR_REG(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature
        self.regularizer_weight = cfg.method_kwargs.regularizer_weight

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # Scheduler params
        self.configure_reg_scheduler(cfg.method_kwargs.reg_scheduler)

        self.manifold_regularizer = ManifoldRegularizer(return_metrics=False)

        try:
            self.layers = cfg.method_kwargs.layers
        except:
            pass

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR_REG, SimCLR_REG).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        cfg.method_kwargs.regularizer_weight = omegaconf_select(
            cfg, "method_kwargs.regularizer_weight", 0.0
        )

        cfg.method_kwargs.layers = omegaconf_select(cfg, "method_kwargs.layers", [])
        cfg.method_kwargs.scheduler = omegaconf_select(
            cfg, "method_kwargs.scheduler", {"name": "constant", "weight": 1.0}
        )
        cfg.method_kwargs.log_images = omegaconf_select(cfg, "method_kwargs.log_images", False)

        return cfg

    def configure_reg_scheduler(self, scheduler):
        if scheduler.name == "triangle":
            self.reg_scheduler = TriangleScheduler(
                start_weight=scheduler.start_weight,
                max_weight=scheduler.max_weight,
                end_weight=scheduler.end_weight,
                start_epoch=scheduler.start_epoch,
                mid_epoch=scheduler.mid_epoch,
                end_epoch=scheduler.end_epoch,
            )
        elif scheduler.name == "warmup":
            self.reg_scheduler = WarmupScheduler(
                base_weight=scheduler.base_weight,
                warmup_epochs=scheduler.warmup_epochs,
                weight=scheduler.weight,
                reg_epochs=scheduler.reg_epochs,
            )
        elif scheduler.name == "step":
            self.reg_scheduler = StepScheduler(
                weight=scheduler.weight,
                steps=scheduler.steps,
                scale=scheduler.scale,
            )
        elif scheduler.name == "interval":
            self.reg_scheduler = IntervalScheduler(
                intervals=scheduler.intervals,
                max_epochs = self.max_epochs,
            ) 
        else:
            self.reg_scheduler = ConstantScheduler(
                weight=scheduler.weight,
            )

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = {}
        handles = []
        if self.training:
            if self.cfg.backbone.name.startswith("resnet"):
                def hook_fn_3(module, input, output):
                    out['layer3'] = nn.functional.adaptive_avg_pool2d(output, (1,1)).flatten(1)
                handle_3 = self.backbone.layer3.register_forward_hook(hook_fn_3)
                handles.append(handle_3)
            else:
                for number, block in enumerate(self.backbone.blocks):
                    def hook_fn(module, input, output, number=number):
                        mean_token_representation = output[:, 1:, :].mean(dim=1)
                        out.update({f"mean_block_{number}": mean_token_representation})
                        out.update({f"cls_block_{number}": output[:, 0, :]})

                    handle = block.register_forward_hook(hook_fn)
                    handles.append(handle)

        out.update(super().forward(X))

        z = self.projector(out["feats"])
        out.update({"z": z})
        for handle in handles:
            handle.remove()
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = {}
        def hook_fn_3(module, input, output):
            out['layer_3'] = output
        handle_3 = self.backbone.layer3.register_forward_hook(hook_fn_3)

        out.update(super().multicrop_forward(X))
        z = self.projector(out["feats"])
        out.update({"z": z})
        handle_3.remove()
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )
        # # ------- manifold regularization -------
        if self.cfg.backbone.name.startswith("resnet"):
            regularizer_loss, metrics = self.manifold_regularizer.manifold_regularizer_loss(
                torch.cat(out['layer3']),
                torch.cat(out['feats'])
            )
        else:
            regularizer_loss=0
            metrics = {}
            last_block_number = len(self.backbone.blocks) - 1
            if last_block_number in self.layers:
                self.layers.remove(last_block_number)
            for layer in self.layers:
                if layer != last_block_number:
                    loss_term, laplacian_metrics = self.manifold_regularizer.manifold_regularizer_loss(
                        out[f"mean_block_{layer}"][0],
                        out[f"mean_block_{last_block_number}"][0],
                    )
                    for key, value in laplacian_metrics.items():
                        metrics[f"{key}_Layer{layer}"] = value
                    regularizer_loss += loss_term
            # Divide loss by the number of terms in the regularization loss
            if len(self.layers) > 0:
                regularizer_loss /= len(self.layers)

        metrics.update({"train_regularization_loss": regularizer_loss})
        regularizer_weight = self.reg_scheduler(self.current_epoch)
        regularization_loss_scaled = regularizer_loss * regularizer_weight
        metrics.update(
            {
                "train_regularizer_weight": regularizer_weight,
                "train_regularization_loss_scaled": regularization_loss_scaled,
            }
        )

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return nce_loss + class_loss + regularization_loss_scaled
