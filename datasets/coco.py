"""
Classes and functions used to create Datasets with COCO format
PyTorch Torch Vision Dataset
https://pytorch.org/vision/0.17/generated/torchvision.datasets.CocoDetection.html
"""

import os
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x
from utils import augmentation_utils, object_detection_utils


class COCO(Dataset):

    def __init__(self, root_dir, d_type, transform):
        self.root_dir = os.path.join(root_dir, d_type)
        self.transform = transform
        self.__prepare_dataset()
    
    def __prepare_dataset(self):
        self.dataset = torchvision.datasets.CocoDetection(
            root = self.root_dir,
            annFile = os.path.join(self.root_dir, "_annotations.coco.json"),
            transforms = self.transforms_func
        )

    def transforms_func(self, image, target):

        width, height = image.size

        boxes = [obj['bbox'] for obj in target]
        labels = [obj['category_id'] for obj in target]

        # Normalize bbox coordinates
        normalized_boxes = [[x / width, y / height, (x + w) / width, (y + h) / height] for x, y, w, h in boxes]

        # Convert to tensors
        boxes_tensor = torch.as_tensor(normalized_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(image)

        return new_image, (boxes_tensor, labels_tensor)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = self.transform(image)
        return image, target


def get_coco_dataset(data, load_train=True, load_test=True):

    (data_dir, args) = data

    # Apply ai8x normalization and transformations for img:
    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = COCO(root_dir=data_dir, d_type='train', transform=transform)
    else:
        train_dataset = None
    
    if load_test:
        test_dataset = COCO(root_dir=data_dir, d_type='test', transform=transform)
    else:
        test_dataset = None
    
    return train_dataset, test_dataset


labels = ('None2', 'bench', 'bicycle', 'bus', 'car', 'cone',
          'fire-hydrant', 'motorcycle', 'pothole', 'powerbox', 
          'stop-sign', 'truck')

datasets = [
    {
        'name': 'coco',
        'input': (3, 128, 128),
        'output': labels,
        'loader': get_coco_dataset,
        'collate': object_detection_utils.collate_fn
    }
]