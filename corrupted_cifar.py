from PIL import Image

import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

from cutout import Cutout


class CustomCIFAR10(datasets.CIFAR10):

    def __init__(self, corruption, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)
        self.reset_hparams()
        if self.train and corruption>0.0:
            for i in range(len(self.targets)):
                self.targets[i] = (self.targets[i] if np.random.random_sample()>corruption else np.random.randint(10))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.train and self.jitter_transform is not None:
            img = self.jitter_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.train and self.cut_transform is not None:
            img = self.cut_transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_hparams(self, hparams):
        if 'bright' in hparams and 'contrast' in hparams and 'sat' in hparams and 'hue' in hparams:
            self.jitter_transform = transforms.ColorJitter(hparams['bright'], hparams['contrast'], hparams['sat'], hparams['hue'])
        if 'cutholes' in hparams and 'cutlength' in hparams:
            self.cut_transform = Cutout(n_holes=int(hparams['cutholes']), length=int(hparams['cutlength']))
    
    def reset_hparams(self):
        self.hparams = None
        self.jitter_transform = None
        self.cut_transform = None