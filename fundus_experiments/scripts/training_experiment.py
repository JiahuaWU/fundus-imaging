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

ex = Experiment('fundus training')
template = "efficientnetb0"
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
    Can generate corruption files from any kind of balance methods by setting format:hdf5 and h5_path
    """

    """------------------------------------------------Dataset Related-----------------------------------------------"""
    root_dir = '/home/jiwu/interpretable-fundus/fundus_experiments/data/fundus_preprocessed_512/train/'
    corruption_path = '/home/jiwu/interpretable-fundus/fundus_experiments/data/corruption/'
    # corruption_dic = {'output_type': 'jpeg', 'ops': [("random", 2)], 'img_size': (512, 512, 3),
    #     #                   'or_less': True, 'valid_corruption': False}
    corruption_dic = None
    if corruption_dic:
        h5_name = 'corruption_' + str(corruption_dic['ops'])
        if corruption_dic['or_less']:
            h5_name += '_or_less_'
        h5_name += root_dir.split('/')[-2].split('_')[-1] + '_'
    dataset = 'trainLabels.csv'
    train_len = None
    valid_len = None
    valid_rate = 0.3
    preprocessing = {'type': 'augmented'}
    gaussian_noise = None
    balance = None
    if corruption_dic:
        if balance:
            h5_name += 'balance'
        else:
            h5_name += 'imbalance'
        h5_name += '.h5'
        h5_path = os.path.join(corruption_path, h5_name)
        corruption_dic['h5_path'] = h5_path

    """-------------------------------------------------Model Related------------------------------------------------"""
    baseline = 'efficientnetb0'
    dropout = None
    adv_training_config = {'type': 'pgd', 'steps': 5, 'epsilon_fgsm': 2.5 / 255., 'batch_acc_tol': 0}
    loss_setting = {'type': 'crossentropy'}
    device = 'cuda:0'
    optim_setting = {'optim': 'adam'}
    weight_dir = None
    parallel = True
    num_epoch = 20
    batch_size = 24
    seed = 19660602
    modelname = 'train_' + baseline + '_' + preprocessing['type'] + '_' + adv_training_config["type"] + '_unfreezed' + \
                '_' + loss_setting['type']
    if weight_dir:
        modelname += '_restart'
    if balance:
        modelname += ('_' + balance)
    if parallel:
        modelname += '_parallel'
    if corruption_dic:
        modelname += '_corrupted'


@ex.automain
def run(_run: sacred.run.Run, modelname, baseline, preprocessing, device, optim_setting, loss_setting, seed,
        train_len, valid_len, valid_rate, num_epoch, batch_size, adv_training_config, root_dir, dataset,
        weight_dir, parallel, balance, gaussian_noise, dropout, corruption_dic):
    # run is a special variable, an instance of https://sacred.readthedocs.io/en/latest/apidoc.html#api-run
    # if we had some input data which is required to run this experiment, we could add it here.
    # This is only important if the input data changes a lot/is generated by some other experiment though
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loaders, datasets, sizes, train_label_dist, valid_label_dist = data.get_fundus_train(root_dir=root_dir,
                                                                                         transform_dic=preprocessing,
                                                                                         original_csv_name=dataset,
                                                                                         seed=seed,
                                                                                         balance=balance,
                                                                                         k=5,
                                                                                         gaussian_noise=gaussian_noise,
                                                                                         shuffle=True,
                                                                                         batch_size=batch_size,
                                                                                         valid_rate=valid_rate,
                                                                                         train_len=train_len,
                                                                                         valid_len=valid_len,
                                                                                         corruption_dic=corruption_dic)
    f = data.FundusDataset(root_dir=root_dir, csv_name=dataset, transform_dic=preprocessing)

    model = get_baseline(baseline, pretrain=True, weights_dir=weight_dir, parallel=parallel, dropout=dropout)
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = model.to(device)

    # Settings
    train_label_dist = train_label_dist.to(device)
    criterion = get_loss(loss_setting, train_label_dist)
    optimizer = get_optimizer(optim_setting, model.parameters())
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160, 180], gamma=0.01)
    print("Training ", modelname)

    model, best_model_weights, optimizer \
        = train_model(model, loaders, sizes, criterion, optimizer, device, scheduler=None, valid=True, ex=ex, seed=seed,
                      fundus_dataset=f, num_epochs=num_epoch, adv_training_config=adv_training_config, return_best=True)

    # this will store any produced artifacts like images, trained weights etc.
    # As well as output the metrics
    torch.save(model.state_dict(), modelname)
    torch.save(best_model_weights, modelname + '_best')
    torch.save(optimizer.state_dict(), 'optimizer_state')
    _run.add_artifact(modelname)
    _run.add_artifact(modelname + '_best')
    _run.add_artifact('optimizer_state')
    # remove artifacts once done to not clutter
    import os
    os.remove(modelname)
    os.remove(modelname + '_best')
    os.remove('optimizer_state')
