import os
import random

from PIL import ImageOps, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder


class ImageFolderWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return index, (*data)


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    def __call__(self, img):
        return ImageOps.solarize(img)


class NCropAugmentation:
    def __init__(self, transform, n_crops=None):
        self.transform = transform

        if isinstance(transform, list):
            self.one_transform_per_crop = True
        else:
            self.one_transform_per_crop = False
            self.n_crops = n_crops

    def __call__(self, x):
        if self.one_transform_per_crop:
            return [transform(x) for transform in self.transform]
        else:
            return [self.transform(x) for i in range(self.n_crops)]


class BaseTransform:
    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        return str(self.transform)


class CifarTransform(BaseTransform):
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )


class STLTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((96, 96), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )


class ImagenetTransform(BaseTransform):
    def __init__(
        self, brightness, contrast, saturation, hue, gaussian_prob=0.5, solarization_prob=0
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)], p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )


class MulticropAugmentation:
    def __init__(
        self, transform, size_crops, n_crops, min_scale_crops, max_scale_crops,
    ):
        self.size_crops = size_crops
        self.n_crops = n_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        self.transforms = []
        for i in range(len(size_crops)):
            rrc = transforms.RandomResizedCrop(
                size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            full_transform = transforms.Compose([rrc, transform])
            self.transforms.append(full_transform)

    def __call__(self, x):
        imgs = []
        for n, transform in zip(self.n_crops, self.transforms):
            imgs.extend([transform(x) for i in range(n)])
        return imgs


class MulticropCifarTransform(BaseTransform):
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )


class MulticropSTLTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )


class MulticropImagenetTransform(BaseTransform):
    def __init__(
        self, brightness, contrast, saturation, hue, gaussian_prob=0.5, solarization_prob=0
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)], p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )


def prepare_transform(dataset, multicrop=False, **kwargs):
    if dataset in ["cifar10", "cifar100"]:
        return CifarTransform() if not multicrop else MulticropCifarTransform()
    elif dataset == "stl10":
        return STLTransform() if not multicrop else MulticropSTLTransform()
    elif dataset in ["imagenet", "imagenet100"]:
        return (
            ImagenetTransform(**kwargs) if not multicrop else MulticropImagenetTransform(**kwargs)
        )


def prepare_n_crop_transform(transform, n_crops=None):
    return NCropAugmentation(transform, n_crops)


def prepare_multicrop_transform(
    transform, size_crops, n_crops=None, min_scale_crops=None, max_scale_crops=None
):
    if n_crops is None:
        n_crops = [2, 6]
    if min_scale_crops is None:
        min_scale_crops = [0.14, 0.05]
    if max_scale_crops is None:
        max_scale_crops = [1.0, 0.14]

    return MulticropAugmentation(
        transform,
        size_crops=size_crops,
        n_crops=n_crops,
        min_scale_crops=min_scale_crops,
        max_scale_crops=max_scale_crops,
    )


def prepare_datasets(
    dataset, data_folder=None, train_dir=None, transform=None, with_index=True,
):
    if data_folder is None:
        if os.path.isdir("/data/datasets"):
            data_folder = "/data/datasets"
        elif os.path.isdir("/datasets"):
            data_folder = "/datasets/"
        else:
            sandbox_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            data_folder = os.path.join(sandbox_folder, "datasets")

    if train_dir is None:
        train_dir = f"{dataset}/train"

    if dataset == "cifar10":
        train_dataset = CIFAR10(
            os.path.join(data_folder, train_dir), train=True, download=True, transform=transform,
        )

    elif dataset == "cifar100":
        train_dataset = CIFAR100(
            os.path.join(data_folder, train_dir), train=True, download=True, transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            os.path.join(data_folder, train_dir),
            split="train+unlabeled",
            download=True,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        train_dir = os.path.join(data_folder, train_dir)
        train_dataset = ImageFolder(train_dir, transform)

    if with_index:
        train_dataset = ImageFolderWithIndex(train_dataset)

    return train_dataset


def prepare_dataloaders(train_dataset, batch_size=64, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
