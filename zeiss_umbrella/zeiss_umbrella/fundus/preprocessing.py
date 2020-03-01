from torchvision import transforms


def normalize():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
                              )


def default():
    return transforms.Compose([transforms.ToTensor()])


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
