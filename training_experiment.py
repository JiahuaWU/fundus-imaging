from zeiss_umbrella.config import FILE_OBSERVER_BASE_PATH, FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH
import sacred
from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from zeiss_umbrella.fundus.setting_parser import get_baseline, get_optimizer, get_loss
from zeiss_umbrella.fundus.train import *
from zeiss_umbrella.fundus import data

"""
Training script using sacred libary which is capable of recording experiment-related
information  automatically.
"""

ex = Experiment('fundus training')
template = ""
ex.observers.append(FileStorageObserver(FILE_OBSERVER_BASE_PATH,
                                        FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH, template))


# uncomment if you use progress bars
# from sacred.utils import apply_backspaces_and_linefeeds
# ex.captured_out_filter = apply_backspaces_and_linefeeds
# for more info see https://sacred.readthedocs.io/en/latest/collected_information.html#live-information


@ex.config
def my_config():
    """
    preprocessing: dictionary, 'type':'centerCrop','default' , 'cropSize':list, 'size':tuple
    adv_training_config: 'type':'baseline', 'fgsm', 'fgsm_k_image', 'pgd', 'boundary_attack'
                     configuration for fgsm type attack:
                     'epsilon_fgsm': maximum pixel-wise amplitude of perturbation default is 1/255
                     'steps' (bim/pgd): number of iterations
                     'alpha_fgsm': step size of each iteration, default is 2.5 * epsilon_fgsm / steps
                     configuration for decision boundary attack (dba):
                     'epsilon_dba': size of orthogonal move of decision boundary attack, default is 1.
                     'delta_dba': size of move towards target sample, default is 0.1
                     'n_step_max_dba': maximum number of iterations for dba, default is 500
                     'e_step_max_dba': maximum number of iterations for epsilon step, default is 20
                     'd_step_max_dba': maximum number of iterations for delta step, default is 10
                     'diff_tol_dba': stop dba if the mean square error is smaller than this value, default is 10
                     'batch_acc_tol': minimum batch accuracy to trigger boundary attack, default is 0.7
                     'unqualified_sample_ratio_tol_dba': return the adversarial if the ratio of the "bad adv samples" is
                                                         smaller than this threshold
                     global configuration:
                     'weight': Weight of adversarial loss in the total loss, default is 0.3
    loss_setting: 'type': 'crossentropy', 'focal_loss', 'class_balanced', 'inv_freq"
                  configuration for 'focal_loss':
                  'gamma': characterize the rate of drop of well classified data
                  'alpha': weights for each class of type torch.tensor
                  configuration for 'class_balanced':
                  'beta'
    """
    baseline = 'resnet50'
    preprocessing = {'type': 'augmented'}
    adv_training_config = {'type': 'baseline'}
    loss_setting = {'type': 'crossentropy'}
    modelname = 'train_' + baseline + '_' + preprocessing['type'] + '_' + adv_training_config["type"] + '_unfreezed' + \
                '_' + loss_setting['type']
    device = 'cuda:0'
    optim_setting = {'optim': 'adam'}
    root_dir = 'fundus/data/fundus_preprocessed_512/train/'
    augmented = False
    train_len = None
    valid_len = None
    valid_rate = 0.3
    num_epoch = 20
    batch_size = 16
    seed = 19660602


@ex.automain
def run(_run: sacred.run.Run, modelname, baseline, preprocessing, device, optim_setting, loss_setting, seed,
        train_len, valid_len, valid_rate, augmented, num_epoch, batch_size, adv_training_config, root_dir):
    loaders, sizes, train_label_dist, valid_label_dist = data.get_fundus_train(root_dir=root_dir,
                                                                               transform_dic=preprocessing,
                                                                               original_csv_name='trainLabels.csv',
                                                                               augmented_csv_name='trainLabels_augmented.csv',
                                                                               augmented=augmented, seed=seed,
                                                                               shuffle=True,
                                                                               batch_size=batch_size,
                                                                               valid_rate=valid_rate,
                                                                               train_len=train_len, valid_len=valid_len)
    f = data.FundusDataset(root_dir=root_dir, csv_name='trainLabels.csv', transform_dic=preprocessing)

    # Freezing the layers to enable fine-tuning only
    # for param in model.parameters():
    #     param.requires_grad = False
    # model = torchvision.models.inception_v3(pretrained=True)
    # # Handle the auxilary net
    # num_ftrs = model.AuxLogits.fc.in_features
    # model.AuxLogits.fc = nn.Linear(num_ftrs, 5)
    model = get_baseline(baseline, pretrain=True)
    # Handle the primary net
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    print(model.__class__.__name__)
    model = model.to(device)

    # Settings
    train_label_dist = train_label_dist.to(device)
    criterion = get_loss(loss_setting, train_label_dist)
    optimizer = get_optimizer(optim_setting, model.parameters())
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160, 180], gamma=0.01)
    print("Training ", modelname)

    # Print number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: ', total_params)
    print('trainable parameters: ', trainable_params)

    model, best_model_weights \
        = train_model(model, loaders, sizes, criterion, optimizer, device, scheduler=None, valid=True, ex=ex,
                      fundus_dataset=f, num_epochs=num_epoch, adv_training_config=adv_training_config, return_best=True)

    # this will store any produced artifacts like images, trained weights etc.
    # As well as output the metrics
    torch.save(model.state_dict(), modelname)
    torch.save(best_model_weights, modelname + '_best')
    _run.add_artifact(modelname)
    _run.add_artifact(modelname + '_best')
    # remove artifacts once done to not clutter
    import os
    os.remove(modelname)
    os.remove(modelname + '_best')
