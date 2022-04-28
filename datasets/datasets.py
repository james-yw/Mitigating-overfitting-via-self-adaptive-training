import os
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 10


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 100


class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform,
                                       target_transform=target_transform, download=download)

        # unify the interface

        import scipy.io as sio

        # reading(loading) mat file as array
        if self.split=='train':
            self.filename = 'train_32x32.mat'
        elif self.split == 'test':
            self.filename = "test_32x32.mat"
        else:
            raise ValueError(f"The {self.split} split operation is NOT Support!")

        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        print("data shape:", self.data.shape)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    @property
    def num_classes(self):
        return 10