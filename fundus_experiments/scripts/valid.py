from zeiss_umbrella.fundus.setting_parser import get_baseline, get_loss
from zeiss_umbrella.fundus.data import get_fundus_train
from zeiss_umbrella.fundus.train import test_model
import torch
import json
import os
from zeiss_umbrella.config import FILE_OBSERVER_BASE_PATH, FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH
import sacred
from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver

ex = Experiment('fundus training')
template = "efficientnetb0"
ex.observers.append(FileStorageObserver(FILE_OBSERVER_BASE_PATH,
                                        FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH, template))


@ex.config
def my_config():
    experiments_path = '/home/jiwu/interpretable-fundus/fundus_experiments'
    exp_dir = 'corruption_experiments/efficientnetb0_corruption_imbalance_3'
    weights_dir = 'corruption_experiments/efficientnetb0_corruption_imbalance_3/train_efficientnetb0_normalize_baseline_unfreezed_crossentropy_parallel_corrupted'
    device = 'cuda:0'
    exp_dir = os.path.join(experiments_path, exp_dir)
    weights_dir = os.path.join(experiments_path, weights_dir)
    with open(os.path.join(exp_dir, 'config.json')) as f:
        config = json.load(f)
    valid_corruption = True


@ex.automain
def run(_run: sacred.run.Run, config, device, valid_corruption):
    seed = config['seed']
    corruption_dic = {
        'h5_path': "/home/jiwu/interpretable-fundus/fundus_experiments/data/corruption/corruption_[('random', 3)]_or_less_512_imbalance.h5"
        , 'valid_corruption': valid_corruption, 'output_type': 'jpeg'}
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loaders, datasets, sizes, train_label_dist, valid_label_dist = get_fundus_train(
        root_dir=os.path.join(experiments_path, config['root_dir']),
        transform_dic=config['preprocessing'],
        original_csv_name=config['dataset'],
        seed=config['seed'],
        balance=config.get('balance', None),
        k=5,
        gaussian_noise=config.get('gaussian_noise', None),
        shuffle=True,
        batch_size=config['batch_size'],
        valid_rate=config['valid_rate'],
        corruption_dic=corruption_dic)
    model = get_baseline(config['baseline'], weights_dir=weights_dir, parallel=config.get('parallel', False))
    model = model.to(device)
    try:
        test_model(model, loaders['valid'], sizes['valid'], get_loss(loss_setting=config['loss_setting']), device,
                   plot_confusion_matrix=True,
                   confusion_matrix_name=weights_dir.split('/')[-2] + '.jpeg', ex=ex)
    except:
        # The loss is not important if we only want to get the confusion matrix
        test_model(model, loaders['valid'], sizes['valid'], get_loss(loss_setting={'type': 'crossentropy'}), device,
                   plot_confusion_matrix=True,
                   confusion_matrix_name=weights_dir.split('/')[-2] + '.jpeg', ex=ex)
