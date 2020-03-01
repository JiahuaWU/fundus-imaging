# boundary attack import
import torch
import numpy as np
import time
import os
import datetime
from zeiss_umbrella.fundus.data import FundusDataset, get_fundus_train
from zeiss_umbrella.fundus.adversarial import generate_target_samples, boundary_attack_image
# load model input
import torch.nn as nn
from zeiss_umbrella.fundus.setting_parser import get_baseline

# For printing the images
import matplotlib.pyplot as plt

# %%
if __name__ == '__main__':
    model = get_baseline('efficientnetb3', weights_dir='runs/36/train_efficientnetb3_augmented_baseline_unfreezed_crossentropy_best',
                         pretrain=True, parallel=True)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    f = FundusDataset(transform_dic={'type': 'default'}, root_dir='data/fundus_preprocessed_512/train/',
                      csv_name='trainLabels.csv')
    input_batch = []
    label_batch = []
    for ind in range(2206, 2214):
        input_batch.append(f[ind][0])
        label_batch.append(f[ind][1])
    label_batch = torch.tensor(label_batch)
    input_batch = torch.stack(input_batch)
    target_samples, target_labels = generate_target_samples(input_batch, label_batch, f, device=device)
    print(label_batch)
    print(target_labels)
    adv_batch, adv_target_batch = boundary_attack_image(model, device, input_batch, label_batch, fundus_dataset=f,
                                                        unqualified_sample_ratio_tol=0, n_step_max=250, diff_tol=10,
                                                        delta=0.1)
