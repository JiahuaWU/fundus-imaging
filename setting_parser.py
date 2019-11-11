from zeiss_umbrella.resnet import *
import torch.optim as optim
import torch.nn as nn
import torch
from .train import FocalLoss_SM


def get_baseline(baseline_name, pretrain=False, progress=True):
    """Parse the input string given in the expriment script and return the corresponding model"""
    if baseline_name == 'resnet18':
        return resnet18(pretrain, progress)
    elif baseline_name == 'resnet34':
        return resnet34(pretrain, progress)
    elif baseline_name == 'resnet50':
        return resnet50(pretrain, progress)
    elif baseline_name == 'resnet101':
        return resnet101(pretrain, progress)
    elif baseline_name == 'resnet152':
        return resnet152(pretrain, progress)


def get_optimizer(optim_setting, model_param):
    if optim_setting['optim'].lower() == 'sgd':
        return optim.SGD(model_param, lr=optim_setting.get('lr', 0.001), momentum=optim_setting.get('momentum', 0.9))
    elif optim_setting['optim'].lower() == 'adam':
        return optim.Adam(model_param, lr=optim_setting.get('lr', 0.001))


def get_loss(loss_setting, distributions, device='cuda'):
    distributions = distributions.type(torch.float)
    if loss_setting['type'] == 'crossentropy':
        return nn.CrossEntropyLoss()
    if loss_setting['type'] == 'class_balanced':
        beta = loss_setting.get('beta', (distributions.sum() - 1.) / distributions.sum())
        weights = (1. - beta) / (1. - torch.pow(beta, distributions.type(torch.float)))
        weights /= weights.sum() * 5
        weights = weights.to(device)
        print(beta)
        print(weights)
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
