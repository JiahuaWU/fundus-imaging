import json
import os
import torch
from torch.utils.data import DataLoader, random_split

import sacred
from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver
from zeiss_umbrella.config import FILE_OBSERVER_BASE_PATH, FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH
from zeiss_umbrella.fundus.adversarial import pgd
from zeiss_umbrella.fundus.data import get_fundus_train
from zeiss_umbrella.fundus.setting_parser import get_baseline, get_loss
from tqdm import tqdm, trange

ex = Experiment('fundus training')
template = "efficientnetb0"
ex.observers.append(FileStorageObserver(FILE_OBSERVER_BASE_PATH,
                                        FILE_OBSERVER_RESOURCE_PATH, FILE_OBSERVER_SOURCE_PATH, template))

@ex.config
def my_config():
    # Loading pretrained models
    experiments_path = '/home/jiwu/interpretable-fundus/fundus_experiments'
    exp_dir = 'baselines/efficientnetb0'
    weights_dir = 'baselines/efficientnetb0/train_efficientnetb0_augmented_baseline_unfreezed_crossentropy_best'
    exp_dir = os.path.join(experiments_path, exp_dir)
    weights_dir = os.path.join(experiments_path, weights_dir)
    device = 'cuda:0'
    with open(os.path.join(exp_dir, 'config.json')) as f:
        config = json.load(f)
    num_images = 240
    init_epsilon = 0
    increment = 0.2 / 255.
    steps = 5


@ex.automain
def run(_run: sacred.run.Run, config, num_images, init_epsilon, increment,
        experiments_path, weights_dir, steps, device):
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    batch_size = int(config['batch_size'] / 4)
    loaders, datasets, sizes, train_label_dist, valid_label_dist = get_fundus_train(
                                                            root_dir=os.path.join(experiments_path, config['root_dir']),
                                                            transform_dic=config['preprocessing'],
                                                            original_csv_name=config['dataset'],
                                                            seed=config['seed'],
                                                            balance=config.get('balance', None),
                                                            k=5,
                                                            gaussian_noise=config.get('gaussian_noise', None),
                                                            shuffle=True,
                                                            batch_size=batch_size,
                                                            valid_rate=config['valid_rate'])
    new_dataset, _ = random_split(datasets['valid'], [num_images, sizes['valid'] - num_images])
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
    model = get_baseline(config['baseline'], weights_dir=weights_dir, parallel=config.get('parallel', False))
    model = model.to(device)
    model.eval()
    criterion = get_loss({'type': 'crossentropy'})
    epsilon = init_epsilon
    for _ in range(3):
        epsilon += increment
        num_samples_passed = 0
        running_incorrects = 0
        batches = tqdm(new_loader, total=num_images / new_loader.batch_size)
        for inputs, labels in batches:
            inputs, labels = inputs.to(device), labels.to(device)
            adv_images = pgd(inputs, labels, model, criterion, device, epsilon=epsilon, steps=steps)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            outputs_adv = model(adv_images)
            _, preds_adv = torch.max(outputs_adv, 1)
            running_incorrects += torch.sum(preds_adv != preds)
            num_samples_passed += batch_size
            if num_samples_passed >= num_images:
                break
        print('running incorrects: {}'.format(running_incorrects))
        print('num_samples_passed: {}'.format(num_samples_passed))
        success_rate = float(running_incorrects) / float(num_samples_passed)
        print('epsilon: {}'.format(epsilon * 255))
        print('success rate: {}'.format(success_rate))
        ex.log_scalar('epsilon', epsilon * 255)
        ex.log_scalar('success rate', success_rate)


