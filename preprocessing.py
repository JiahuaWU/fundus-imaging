from torchvision import transforms
import torch
import cv2

"""
On-fly preprocessing
"""

def default_transform():
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
                              )


def augmented():
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.RandomAffine(degrees=(-30, 10), translate=(0, 0.05), scale=(0.9, 1.1),
                                                       shear=10,
                                                       fillcolor=(128, 128, 128)),
                               transforms.ColorJitter(brightness=0.15, contrast=0.15),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
                              )


def resize(size):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(size),
                               transforms.ToTensor()])


def center_crop(size):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.CenterCrop(size=size),
                               # The dataset only considers the 32 most centered pixels
                               transforms.ToTensor()
                               ])
