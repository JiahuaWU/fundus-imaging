from zeiss_umbrella.resnet import *
import torch.optim as optim
import torch.nn as nn
import torch
from .train import FocalLoss_SM
from torchvision.models import inception_v3
from efficientnet_pytorch import EfficientNet


def get_baseline(baseline_name, num_classes=5, pretrain=False, progress=True,
                 fine_tune=False, weights_dir=None, parallel=False, dropout=None):
    """Parse the input string given in the expriment script and return the corresponding model"""
    if baseline_name == 'resnet18':
        model = resnet18(pretrain, progress, dropout=dropout)
    elif baseline_name == 'resnet34':
        model = resnet34(pretrain, progress, dropout=dropout)
    elif baseline_name == 'resnet50':
        model = resnet50(pretrain, progress, dropout=dropout)
    elif baseline_name == 'resnet101':
        model = resnet101(pretrain, progress, dropout=dropout)
    elif baseline_name == 'resnet152':
        model = resnet152(pretrain, progress, dropout=dropout)
    elif baseline_name == 'inceptionv3':
        model = inception_v3(pretrain, progress)
    elif baseline_name[:-2] == 'efficientnet':
        version = baseline_name[-1]
        model = EfficientNet.from_pretrained('efficientnet-b' + version, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError('Net not supported')
    trained_with_parallel = False
    if weights_dir is not None and 'parallel' in weights_dir:
        trained_with_parallel = True
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    if model.__class__.__name__ != 'EfficientNet':
        if model.__class__.__name__ == 'Inception3':
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    if weights_dir:
        if trained_with_parallel:
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(weights_dir))
    if parallel and not trained_with_parallel:
        model = nn.DataParallel(model)
    # Output model information
    print(model.__class__.__name__)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: ', total_params)
    print('trainable parameters: ', trainable_params)
    return model


def get_optimizer(optim_setting, model_param):
    state_dir = optim_setting.get('state_dir', None)
    if optim_setting['optim'].lower() == 'sgd':
        optimizer = optim.SGD(model_param, lr=optim_setting.get('lr', 0.001),
                              momentum=optim_setting.get('momentum', 0.9))
    elif optim_setting['optim'].lower() == 'adam':
        optimizer = optim.Adam(model_param, lr=optim_setting.get('lr', 0.001))
    if state_dir:
        optimizer.load_state_dict(torch.load(state_dir))
    return optimizer


def get_loss(loss_setting=None, distributions=None, device='cuda'):
    """
    return loss functions. Supported loss function: crossentropy, focal loss(https://arxiv.org/abs/1708.02002),
    inverse frequency loss, class balanced loss(https://arxiv.org/abs/1901.05555)
    """
    if distributions.__class__.__name__ != 'NoneType':
        distributions = distributions.type(torch.float)
    if loss_setting['type'] == 'crossentropy' or loss_setting is None:
        return nn.CrossEntropyLoss()
    if loss_setting['type'] == 'class_balanced':
        beta = loss_setting.get('beta', (distributions.sum() - 1.) / distributions.sum())
        weights = (1. - beta) / (1. - torch.pow(beta, distributions.type(torch.float)))
        weights /= weights.sum() * 5
        weights = weights.to(device)
        print("beta used: {}".format(beta))
        print("weights of classes: {}".format(weights))
        return nn.CrossEntropyLoss(weights)
    if loss_setting['type'] == 'focal_loss':
        gamma = loss_setting.get('gamma', 2)
        return FocalLoss_SM(gamma)
    if loss_setting['type'] == 'inv_freq':
        weights = 1. / distributions
        return nn.CrossEntropyLoss(weights / weights.sum())
    if loss_setting['type'] == 'focal_inv':
        gamma = loss_setting.get('gamma', 2)
        weights = 1. / distributions
        return FocalLoss_SM(gamma=gamma, alpha=weights / weights.sum())
