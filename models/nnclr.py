import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.nnclr import nnclr_loss_func
from utils.metrics import accuracy_at_k
from utils.gather_layer import gather


class NNCLR(Model):
    def __init__(self, args):
        super().__init__(args)

        proj_hidden_dim = args.hidden_dim
        output_dim = args.encoding_dim
        pred_hidden_dim = args.pred_hidden_dim
        assert output_dim > 0

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # queue
        self.queue_size = args.queue_size
        self.register_buffer("queue", torch.randn(self.queue_size, output_dim))
        self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, z, y):
        z = gather(z)
        y = gather(y)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        self.queue_y[ptr : ptr + batch_size] = y
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def find_nn(self, z):
        idx = torch.cdist(z, self.queue).min(dim=1)[1]
        nn = self.queue[idx]
        return idx, nn

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projector(features)
            p = self.predictor(z)
            return z, p, y

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # forward online encoder
        z1, p1, output1 = self(X1, classify_only=False)
        z2, p2, output2 = self(X2, classify_only=False)

        # find nn
        idx1, nn1 = self.find_nn(z1)
        idx2, nn2 = self.find_nn(z2)

        # compute nn accuracy
        b = target.size(0)
        nn_acc = (target == self.queue_y[idx1]).sum() / b

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1, target)

        # ------- contrastive loss -------
        nnclr_loss = nnclr_loss_func(nn1, p2) / 2 + nnclr_loss_func(nn2, p1) / 2

        # ------- classification loss -------
        output = torch.cat((output1, output2))
        target = target.repeat(2)

        # ------- classification loss -------
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = nnclr_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_nnclr_loss": nnclr_loss,
            "train_class_loss": class_loss,
            "train_nn_acc": nn_acc,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss
