############### configuration file ###############
import numpy as np

import torch
import torchvision.transforms as transforms
import utils.mytransforms as mytransforms

#- Augmentation -#
train_transform = {
            'cifar10': 
            transforms.Compose([
                transforms.RandomCrop(32, padding=4, fill=tuple([int(255*x) for x in mytransforms.CIFAR10_STATS['mean']])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.CIFAR10_STATS['mean'], mytransforms.CIFAR10_STATS['std']),
            ]),
            'cifar100':
            transforms.Compose([
                transforms.RandomCrop(32, padding=4, fill=tuple([int(255*x) for x in mytransforms.CIFAR100_STATS['mean']])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.CIFAR100_STATS['mean'], mytransforms.CIFAR100_STATS['std']),
            ])
}

test_transform = {
            'cifar10':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.CIFAR10_STATS['mean'], mytransforms.CIFAR10_STATS['std']),
            ]),
            'cifar100':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.CIFAR100_STATS['mean'], mytransforms.CIFAR100_STATS['std']),
            ])
}

#----------13-layer Net-------------#
layer13 = {
    'batch_size' : 128,
    'lrs' : np.logspace(-1, -4, 50, base=10, dtype='float32'),
    'weight_decay' : 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'loss': {'name': 'Softmax'}
}

layer13_largemargin = {
    'batch_size' : 128,
    'lrs' : np.logspace(-1, -4, 50, base=10, dtype='float32'),
    'weight_decay' : 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'loss': {'name': 'LargeMarginInSoftmax', 'reg_lambda':0.3}
}