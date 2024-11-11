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

import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Union
from subprocess import call

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from solo.utils.misc import omegaconf_select
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class Mux:
    def __init__(self, prob: float):
        """Implements mutex operation for dali in order to support probabilitic augmentations.

        Args:
            prob (float): probability value
        """

        self.to_bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, true_case, false_case):
        condition = self.to_bool(self.rng())
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case


class RandomGrayScaleConversion:
    def __init__(self, prob: float = 0.2, device: str = "gpu"):
        """Converts image to greyscale with probability.

        Args:
            prob (float, optional): probability of conversion. Defaults to 0.2.
            device (str, optional): device on which the operation will be performed.
                Defaults to "gpu".
        """

        self.mux = Mux(prob=prob)
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        out = self.grayscale(images)
        out = fn.cat(out, out, out, axis=2)
        return self.mux(true_case=out, false_case=images)


class RandomColorJitter:
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        prob: float = 0.8,
        device: str = "gpu",
    ):
        """Applies random color jittering with probability.

        Args:
            brightness (float): brightness value for samplying uniformly
                in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): contrast value for samplying uniformly
                in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): saturation value for samplying uniformly
                in [max(0, 1 - saturation), 1 + saturation].
            hue (float): hue value for samplying uniformly in [-hue, hue].
            prob (float, optional): probability of applying jitter. Defaults to 0.8.
            device (str, optional): device on which the operation will be performed.
                Defaults to "gpu".
        """

        assert 0 <= hue <= 0.5

        self.mux = Mux(prob=prob)

        self.color = ops.ColorTwist(device=device)

        # look at torchvision docs to see how colorjitter samples stuff
        # for bright, cont and sat, it samples from [1-v, 1+v]
        # for hue, it samples from [-hue, hue]

        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        if brightness:
            self.brightness = ops.random.Uniform(range=[max(0, 1 - brightness), 1 + brightness])

        if contrast:
            self.contrast = ops.random.Uniform(range=[max(0, 1 - contrast), 1 + contrast])

        if saturation:
            self.saturation = ops.random.Uniform(range=[max(0, 1 - saturation), 1 + saturation])

        if hue:
            # dali uses hue in degrees for some reason...
            hue = 360 * hue
            self.hue = ops.random.Uniform(range=[-hue, hue])

    def __call__(self, images):
        out = self.color(
            images,
            brightness=self.brightness() if callable(self.brightness) else self.brightness,
            contrast=self.contrast() if callable(self.contrast) else self.contrast,
            saturation=self.saturation() if callable(self.saturation) else self.saturation,
            hue=self.hue() if callable(self.hue) else self.hue,
        )
        return self.mux(true_case=out, false_case=images)


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.5, window_size: int = 23, device: str = "gpu"):
        """Applies random gaussian blur with probability.

        Args:
            prob (float, optional): probability of applying random gaussian blur. Defaults to 0.5.
            window_size (int, optional): window size for gaussian blur. Defaults to 23.
            device (str, optional): device on which the operation will be performe.
                Defaults to "gpu".
        """

        self.mux = Mux(prob=prob)
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(window_size, window_size))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        sigma = self.sigma() * 1.9 + 0.1
        out = self.gaussian_blur(images, sigma=sigma)
        return self.mux(true_case=out, false_case=images)


class RandomSolarize:
    def __init__(self, threshold: int = 128, prob: float = 0.0):
        """Applies random solarization with probability.

        Args:
            threshold (int, optional): threshold for inversion. Defaults to 128.
            prob (float, optional): probability of solarization. Defaults to 0.0.
        """

        self.mux = Mux(prob=prob)

        self.threshold = threshold

    def __call__(self, images):
        inverted_img = 255 - images
        mask = images >= self.threshold
        out = mask * inverted_img + (True ^ mask) * images
        return self.mux(true_case=out, false_case=images)


class NormalPipelineBuilder:
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        validation: bool = False,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
        data_fraction: float = -1.0,
        use_tfrecords: bool = False
    ):
        """Initializes the pipeline for validation or linear eval training.

        If validation is set to True then images will only be resized to 256px and center cropped
        to 224px, otherwise random resized crop, horizontal flip are applied. In both cases images
        are normalized.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            validation (bool): whether it is validation or training. Defaults to False. Defaults to
                False.
            device_id (int): id of the device used to initialize the seed and for parent class.
                Defaults to 0.
            shard_id (int): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int): total number of shards. Defaults to 1.
            num_threads (int): number of threads to run in parallel. Defaults to 4.
            seed (int): seed for random number generation. Defaults to 12.
            data_fraction (float): percentage of data to use. Use all data when set to -1.0.
                Defaults to -1.0.
        """

        super().__init__()

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed + device_id

        self.device = device
        self.validation = validation

        self.use_tfrecords = use_tfrecords

        # manually load files and labels
        labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())
        data = [
            (data_path / label / file, label_idx)
            for label_idx, label in enumerate(labels)
            for file in sorted(os.listdir(data_path / label))
        ]
        files, labels = map(list, zip(*data))

        # sample data if needed
        if data_fraction > 0:
            assert data_fraction < 1, "data_fraction must be smaller than 1."

            from sklearn.model_selection import train_test_split

            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
        if self.use_tfrecords:
            print("Using TFRecord reader")

            tf_files = sorted(os.listdir(os.path.join(data_path, "data")))

            # Create dir for idx files if not exists.
            idx_files_dir = os.path.join(data_path, "idx_files")
            if not os.path.exists(idx_files_dir):
                os.mkdir(idx_files_dir)

            tfrec_path_list = []
            idx_path_list = []
            n_samples = 0
            # Create idx files and create TFRecordPipelines.
            for tf_file in tf_files:
                # Path of tf_file and idx file.
                tfrec_path = os.path.join(data_path, "data", tf_file)
                tfrec_path_list.append(tfrec_path)
                idx_path = os.path.join(idx_files_dir, tf_file + "_idx")
                idx_path_list.append(idx_path)
                # Create idx file for tf_file by calling tfrecord2idx script.
                if not os.path.isfile(idx_path):
                    call(["tfrecord2idx", tfrec_path, idx_path])
                with open(idx_path, "r") as f:
                    n_samples += len(f.readlines())

            self.reader = fn.readers.tfrecord(
                path=tfrec_path_list,
                index_path=idx_path_list,
                features={
                    "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),  # type: ignore
                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),  # type: ignore
                },
                num_shards=num_shards,
                shard_id=shard_id,
                name="Reader",
            )
        else:
            self.reader = ops.readers.File(
                files=files,
                labels=labels,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=not self.validation,
            )

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        # crop operations
        if self.validation:
            self.resize = ops.Resize(
                device=self.device,
                resize_shorter=256,
                interp_type=types.INTERP_CUBIC,
            )
            # center crop and normalize
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[v * 255 for v in IMAGENET_DEFAULT_MEAN],
                std=[v * 255 for v in IMAGENET_DEFAULT_STD],
            )
        else:
            self.resize = ops.RandomResizedCrop(
                device=self.device,
                size=224,
                random_area=(0.08, 1.0),
                interp_type=types.INTERP_CUBIC,
            )
            # normalize and horizontal flip
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[v * 255 for v in IMAGENET_DEFAULT_MEAN],
                std=[v * 255 for v in IMAGENET_DEFAULT_STD],
            )

        self.coin05 = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    @pipeline_def
    def pipeline(self):
        """Defines the computational pipeline for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        images = self.resize(images)

        if self.validation:
            # crop and normalize
            images = self.cmn(images)
        else:
            # normalize and maybe apply horizontal flip with 0.5 chance
            images = self.cmn(images, mirror=self.coin05())

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (images, labels)


class CustomNormalPipelineBuilder(NormalPipelineBuilder):
    """Initializes the custom pipeline for validation or linear eval training.
    This acts as a placeholder and behaves exactly like NormalPipeline.
    If you want to do exoteric augmentations, you can just re-write this class.
    """

    pass


def build_transform_pipeline_dali(dataset, cfg, dali_device):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet75": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet50": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet25": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    augmentations = []
    if cfg.rrc.enabled:
        augmentations.append(
            ops.RandomResizedCrop(
                device=dali_device,
                size=cfg.crop_size,
                random_area=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                interp_type=types.INTERP_CUBIC,
            )
        )
    else:
        augmentations.append(
            ops.Resize(
                device=dali_device,
                size=(cfg.crop_size, cfg.crop_size),
                interp_type=types.INTERP_CUBIC,
            )
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            RandomColorJitter(
                brightness=cfg.color_jitter.brightness,
                contrast=cfg.color_jitter.contrast,
                saturation=cfg.color_jitter.saturation,
                hue=cfg.color_jitter.hue,
                prob=cfg.color_jitter.prob,
                device=dali_device,
            )
        )

    if cfg.grayscale.prob:
        augmentations.append(RandomGrayScaleConversion(prob=cfg.grayscale.prob, device=dali_device))

    if cfg.gaussian_blur.prob:
        augmentations.append(RandomGaussianBlur(prob=cfg.gaussian_blur.prob, device=dali_device))

    if cfg.solarization.prob:
        augmentations.append(RandomSolarize(prob=cfg.solarization.prob))

    if cfg.equalization.prob:
        raise NotImplementedError(
            "Equalization is not available for DALI"
            "turn it off by setting augmentations.equalization.enabled to False."
        )

    coin = None
    if cfg.horizontal_flip.prob:
        coin = ops.random.CoinFlip(probability=cfg.horizontal_flip.prob)

    cmn = ops.CropMirrorNormalize(
        device=dali_device,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mean=[v * 255 for v in mean],
        std=[v * 255 for v in std],
    )

    class AugWrapper:
        def __init__(self, augmentations, cmn, coin) -> None:
            self.augmentations = augmentations
            self.cmn = cmn
            self.coin = coin

        def __call__(self, images):
            for aug in self.augmentations:
                images = aug(images)
            if self.coin:
                images = self.cmn(images, mirror=self.coin())
            else:
                images = self.cmn(images)
            return images

        def __repr__(self) -> str:
            return f"{self.augmentations}"

    return AugWrapper(augmentations=augmentations, cmn=cmn, coin=coin)


class PretrainPipelineBuilder:
    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int,
        device: str,
        transforms: Callable,
        random_shuffle: bool = True,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
        no_labels: bool = False,
        encode_indexes_into_labels: bool = False,
        data_fraction: float = -1.0,
        use_tfrecords: bool = False
    ):
        """Builder for a pretrain pipeline with Nvidia DALI.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            transforms (Callable): list of transformations.
            num_crops_per_aug (List[int]): number of crops per pipeline.
            random_shuffle (bool, optional): whether to randomly shuffle the samples.
                Defaults to True.
            device_id (int, optional): id of the device used to initialize the seed and
                for parent class. Defaults to 0.
            shard_id (int, optional): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int, optional): total number of shards. Defaults to 1.
            num_threads (int, optional): number of threads to run in parallel. Defaults to 4.
            seed (int, optional): seed for random number generation. Defaults to 12.
            no_labels (bool, optional): if the data has no labels. Defaults to False.
            encode_indexes_into_labels (bool, optional): uses sample indexes as labels
                and then gets the labels from a lookup table. This may use more CPU memory,
                so just use when needed. Defaults to False.
            data_fraction (float): percentage of data to use. Use all data when set to -1.
                Defaults to -1.
        """

        super().__init__()

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed + device_id
        self.use_tfrecords = use_tfrecords

        self.device = device

        data_path = Path(data_path)

        # manually load files and labels
        if no_labels:
            files = [data_path / f for f in sorted(os.listdir(data_path))]
            labels = [-1] * len(files)
        else:
            labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())
            data = [
                (data_path / label / file, label_idx)
                for label_idx, label in enumerate(labels)
                for file in sorted(os.listdir(data_path / label))
            ]
            files, labels = map(list, zip(*data))

        if data_fraction > 0:
            assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
            print(f"Using a fraction of {data_fraction} for the train_size.")
            if no_labels:
                labels = [-1] * len(files)
            else:
                labels = [l for _, l in data]

            from sklearn.model_selection import train_test_split

            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            self.reader = ops.readers.File(
                files=files,
                labels=labels,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
            )

        if encode_indexes_into_labels:
            encoded_labels = []

            self.conversion_map = []
            for file_idx, label_idx in enumerate(labels):
                encoded_labels.append(file_idx)
                self.conversion_map.append(label_idx)

            # to assert that everything is fine
            for file_idx, label_idx in zip(encoded_labels, labels):
                assert self.conversion_map[file_idx] == label_idx

            # use the encoded labels which will be decoded later
            labels = encoded_labels

        if self.use_tfrecords:
            print("Using TFRecord reader")

            tf_files = sorted(os.listdir(os.path.join(data_path, "data")))

            # Create dir for idx files if not exists.
            idx_files_dir = os.path.join(data_path, "idx_files")
            if not os.path.exists(idx_files_dir):
                os.mkdir(idx_files_dir)

            tfrec_path_list = []
            idx_path_list = []
            n_samples = 0
            # Create idx files and create TFRecordPipelines.
            for tf_file in tf_files:
                # Path of tf_file and idx file.
                tfrec_path = os.path.join(data_path, "data", tf_file)
                tfrec_path_list.append(tfrec_path)
                idx_path = os.path.join(idx_files_dir, tf_file + "_idx")
                idx_path_list.append(idx_path)
                # Create idx file for tf_file by calling tfrecord2idx script.
                if not os.path.isfile(idx_path):
                    call(["tfrecord2idx", tfrec_path, idx_path])
                with open(idx_path, "r") as f:
                    n_samples += len(f.readlines())


            self.reader = fn.readers.tfrecord(
                path=tfrec_path_list,
                index_path=idx_path_list,
                features={
                    "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),  # type: ignore
                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),  # type: ignore
                },
                num_shards=num_shards,
                shard_id=shard_id,
                name="Reader",
                random_shuffle=True,
            )
        else:
            self.reader = ops.readers.File(
                files=files,
                labels=labels,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
            )   

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

        self.transforms = transforms

    @pipeline_def
    def pipeline(self):
        """Defines the computational pipeline for dali operations."""

        # read images from memory
        if self.use_tfrecords:
            inputs = self.reader['image/encoded']
            labels = self.reader["image/class/label"] - 1
        else:
            inputs, labels = self.reader(name="Reader")

        images = self.decode(inputs)

        crops = self.transforms(images)

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (*crops, labels)

    def __repr__(self) -> str:
        return str(self.transforms)


class BaseWrapper(DALIGenericIterator):
    """Temporary fix to handle LastBatchPolicy.DROP."""

    def __len__(self):
        size = (
            self._size_no_pad // self._shards_num
            if self._last_batch_policy == LastBatchPolicy.DROP
            else self.size
        )
        if self._reader_name:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / self.batch_size)

            return size // self.batch_size
        else:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / (self._devices * self.batch_size))

            return size // (self._devices * self.batch_size)


class PretrainWrapper(BaseWrapper):
    def __init__(
        self,
        model_batch_size: int,
        model_rank: int,
        model_device: str,
        dataset_size: int,
        conversion_map: List[int] = None,
        *args,
        **kwargs,
    ):
        """Adds indices to a batch fetched from the parent.

        Args:
            model_batch_size (int): batch size.
            model_rank (int): rank of the current process.
            model_device (str): id of the current device.
            dataset_size (int): number of samples in the dataset.
            conversion_map  (List[int], optional): list of integers that map each index
                to a class label. If nothing is passed, no label mapping needs to be done.
                Defaults to None.
        """

        super().__init__(*args, **kwargs)
        self.model_batch_size = model_batch_size
        self.model_rank = model_rank
        self.model_device = model_device
        self.dataset_size = dataset_size
        self.conversion_map = conversion_map
        if self.conversion_map is not None:
            self.conversion_map = torch.tensor(
                self.conversion_map, dtype=torch.float32, device=self.model_device
            ).reshape(-1, 1)
            self.conversion_map = nn.Embedding.from_pretrained(self.conversion_map)

    def __next__(self):
        batch = super().__next__()[0]
        # PyTorch Lightning does double buffering
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1316,
        # and as DALI owns the tensors it returns the content of it is trashed so the copy needs,
        # to be made before returning.

        if self.conversion_map is not None:
            *all_X, indexes = [batch[v] for v in self.output_map]
            targets = self.conversion_map(indexes).flatten().long().detach().clone()
            indexes = indexes.flatten().long().detach().clone()
        else:
            *all_X, targets = [batch[v] for v in self.output_map]
            targets = targets.squeeze(-1).long().detach().clone()
            # creates dummy indexes
            indexes = (
                (
                    torch.arange(self.model_batch_size, device=self.model_device)
                    + (self.model_rank * self.model_batch_size)
                )
                .detach()
                .clone()
            )

        all_X = [x.detach().clone() for x in all_X]
        return [indexes, all_X, targets]


class Wrapper(BaseWrapper):
    def __init__(self, dataset_size: int, *args, **kwargs):
        """Wrapper to have dataset size.

        Args:
            dataset_size (int): number of samples in the dataset.
        """

        super().__init__(*args, **kwargs)
        self.dataset_size = dataset_size

    def __next__(self):
        batch = super().__next__()
        x, target = batch[0]["x"], batch[0]["label"]
        target = target.squeeze(-1).long()
        # PyTorch Lightning does double buffering
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1316,
        # and as DALI owns the tensors it returns the content of it is trashed so the copy needs,
        # to be made before returning.
        x = x.detach().clone()
        target = target.detach().clone()
        return x, target


class PretrainDALIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        train_data_path: Union[str, Path],
        transforms: List[Callable],
        num_large_crops: int,
        num_small_crops: int,
        batch_size: int,
        num_workers: int = 4,
        no_labels=False,
        data_fraction: float = -1.0,
        dali_device: str = "gpu",
        encode_indexes_into_labels: bool = False,
        use_tfrecords: bool = False
    ):
        """DataModule for pretrain data using Nvidia DALI.

        Args:
            dataset (str): dataset name.
            train_data_path (Union[str, Path]): path where the training data is located.
            unique_augs (int): number of unique augmentation pielines
            transforms (List[Callable]): list of transformations.
            num_crops_per_aug (List[int]): number of crops per pipeline.
            num_large_crops (int): total number of large crops.
            num_small_crops (int): total number of small crops.
            batch_size (int): batch size..
            num_workers (int, optional): number of parallel workers. Defaults to 4.
            data_fraction (Optional[float]): percentage of data to use.
                Use all data when set to -1.0. Defaults to -1.0.
            dali_device (str, optional): device used by the dali pipeline.
                Either 'gpu' or 'cpu'. Defaults to 'gpu'.
            encode_indexes_into_labels (bool, optional). Encodes instance indexes
                together with labels. Allows user to access the true instance index.
                Defaults to False.

        """

        super().__init__()

        self.dataset = dataset

        # paths
        self.train_data_path = Path(train_data_path)

        # augmentation-related
        self.transforms = transforms
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops

        self.num_workers = num_workers

        self.batch_size = batch_size

        self.no_labels = no_labels
        self.data_fraction = data_fraction

        self.dali_device = dali_device
        assert dali_device in ["gpu", "cpu"]
        # hack to encode image indexes into the labels
        self.encode_indexes_into_labels = encode_indexes_into_labels

        self.use_tfrecords = use_tfrecords

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.dali = omegaconf_select(cfg, "dali", {})
        cfg.dali.device = omegaconf_select(cfg, "dali.device", "gpu")
        cfg.dali.encode_indexes_into_labels = omegaconf_select(
            cfg, "dali.encode_indexes_into_labels", False
        )
        return cfg

    def setup(self, stage: Optional[str] = None):
        # extra info about training
        self.device_id = self.trainer.local_rank
        self.shard_id = self.trainer.global_rank
        self.num_shards = self.trainer.world_size

        # get current device
        if torch.cuda.is_available() and self.dali_device == "gpu":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

    def train_dataloader(self):
        train_pipeline_builder = PretrainPipelineBuilder(
            self.train_data_path,
            batch_size=self.batch_size,
            transforms=self.transforms,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
            no_labels=self.no_labels,
            encode_indexes_into_labels=self.encode_indexes_into_labels,
            data_fraction=self.data_fraction,
            use_tfrecords = self.use_tfrecords
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        output_map = (
            [f"large{i}" for i in range(self.num_large_crops)]
            + [f"small{i}" for i in range(self.num_small_crops)]
            + ["label"]
        )

        policy = LastBatchPolicy.DROP
        conversion_map = (
            train_pipeline_builder.conversion_map if self.encode_indexes_into_labels else None
        )
        train_loader = PretrainWrapper(
            model_batch_size=self.batch_size,
            model_rank=self.device_id,
            model_device=self.device,
            dataset_size=train_pipeline.epoch_size("Reader"),
            conversion_map=conversion_map,
            pipelines=train_pipeline,
            output_map=output_map,
            reader_name="Reader",
            last_batch_policy=policy,
            auto_reset=True,
        )

        return train_loader


class ClassificationDALIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        train_data_path: Union[str, Path],
        val_data_path: Union[str, Path],
        batch_size: int,
        num_workers: int = 4,
        data_fraction: float = -1.0,
        dali_device: str = "gpu",
        use_tfrecords: bool = False
    ):
        """DataModule for classification data using Nvidia DALI.

        Args:
            dataset (str): dataset name.
            train_data_path (Union[str, Path]): path where the training data is located.
            val_data_path (Union[str, Path]): path where the validation data is located.
            batch_size (int): batch size..
            num_workers (int, optional): number of parallel workers. Defaults to 4.
            data_fraction (float, optional): percentage of data to use.
                Use all data when set to -1.0. Defaults to -1.0.
            dali_device (str, optional): device used by the dali pipeline.
                Either 'gpu' or 'cpu'. Defaults to 'gpu'.
        """

        super().__init__()

        self.dataset = dataset

        # paths
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)

        self.num_workers = num_workers

        self.batch_size = batch_size

        self.data_fraction = data_fraction

        self.dali_device = dali_device
        assert dali_device in ["gpu", "cpu"]

        self.use_tfrecords = use_tfrecords

        # handle custom data by creating the needed pipeline
        if dataset in ["imagenet100", "imagenet75", "imagenet50", "imagenet25", "imagenet"]:
            self.pipeline_class = NormalPipelineBuilder
        elif dataset == "custom":
            self.pipeline_class = CustomNormalPipelineBuilder
        else:
            raise ValueError(dataset, "is not supported, used [imagenet, imagenet100 or custom]")

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.dali = omegaconf_select(cfg, "dali", {})
        cfg.dali.device = omegaconf_select(cfg, "dali.device", "gpu")
        return cfg

    def setup(self, stage: Optional[str] = None):
        # extra info about training
        if not self.trainer:
            self.trainer = pl.Trainer()
        self.device_id = self.trainer.local_rank
        self.shard_id = self.trainer.global_rank
        self.num_shards = self.trainer.world_size

        # get current device
        if torch.cuda.is_available() and self.dali_device == "gpu":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

    def train_dataloader(self):
        train_pipeline_builder = self.pipeline_class(
            self.train_data_path,
            validation=False,
            batch_size=self.batch_size,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
            data_fraction=self.data_fraction,
            use_tfrecords = self.use_tfrecords
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        train_loader = Wrapper(
            pipelines=train_pipeline,
            dataset_size=train_pipeline.epoch_size("Reader"),
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )

        return train_loader

    def val_dataloader(self) -> DALIGenericIterator:
        val_pipeline_builder = self.pipeline_class(
            self.val_data_path,
            validation=True,
            batch_size=self.batch_size,
            device=self.dali_device,
            device_id=self.device_id,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            num_threads=self.num_workers,
        )
        val_pipeline = val_pipeline_builder.pipeline(
            batch_size=val_pipeline_builder.batch_size,
            num_threads=val_pipeline_builder.num_threads,
            device_id=val_pipeline_builder.device_id,
            seed=val_pipeline_builder.seed,
        )
        val_pipeline.build()

        val_loader = Wrapper(
            pipelines=val_pipeline,
            dataset_size=val_pipeline.epoch_size("Reader"),
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        return val_loader
