"""
Adapted from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.
"""
from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import tqdm

from torchvision.datasets.folder import has_file_allowed_extension, is_image_file, make_dataset, pil_loader, accimage_loader, default_loader, IMG_EXTENSIONS

class RAMDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    
    This data loader loads all samples into memory on initialization, instead
    of loading samples on iteration over the dataset.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples.
        images (list): List of pre-loaded PIL images.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(RAMDatasetFolder, self).__init__(root, transform=transform,
                                               target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        # Load all samples into RAM at initialization time
        self.load_all()


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def load_all(self):
        """
        Preload all images without transformations into memory.
        Transformations are to be applied when accessing the dataset.

        Throws:
            MemoryError: when the dataset cannot fit in memory.
        """
        self.images = []
        try:
            for sample in tqdm.tqdm(self.samples, desc='Pre-loading dataset'):
                path, _ = sample
                self.images.append(self.loader(path))
        except MemoryError:
            raise MemoryError("Dataset cannot fit in memory! Please run "
                              "without --ram-dataset.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.images[index]
        _, target = self.samples[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)



class RAMImageFolder(RAMDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    This data loader loads all samples into memory on initialization, instead
    of loading samples on iteration over the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples.
        loaded_imgs (list): List of pre-loaded PIL images.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(RAMImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                             transform=transform,
                                             target_transform=target_transform,
                                             is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.loaded_imgs = self.images
