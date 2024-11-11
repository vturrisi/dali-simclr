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

from solo.methods.barlow_twins import BarlowTwins
from solo.methods.base import BaseMethod
from solo.methods.byol import BYOL
from solo.methods.deepclusterv2 import DeepClusterV2
from solo.methods.dino import DINO
from solo.methods.linear import LinearModel
from solo.methods.mae import MAE
from solo.methods.mocov2plus import MoCoV2Plus
from solo.methods.mocov3 import MoCoV3
from solo.methods.nnbyol import NNBYOL
from solo.methods.nnclr import NNCLR
from solo.methods.nnsiam import NNSiam
from solo.methods.ressl import ReSSL
from solo.methods.simclr import SimCLR
from solo.methods.simsiam import SimSiam
from solo.methods.supcon import SupCon
from solo.methods.swav import SwAV
from solo.methods.vibcreg import VIbCReg
from solo.methods.vicreg import VICReg
from solo.methods.wmse import WMSE
from solo.methods.mae_regularized import MAE_REG
from solo.methods.simclr_regularized import SimCLR_REG
from solo.methods.vicreg_regularized import VICReg_REG
from solo.methods.u_mae import U_MAE

METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mae": MAE,
    "mocov2plus": MoCoV2Plus,
    "mocov3": MoCoV3,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
    "mae-reg": MAE_REG,
    "u-mae": U_MAE,
    "simclr-reg": SimCLR_REG,
    "vicreg-reg": VICReg_REG,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "MAE",
    "LinearModel",
    "MoCoV2Plus",
    "MoCoV3",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
    "MAE_REG",
    "SimCLR_REG",
    "VICReg_REG",
]
