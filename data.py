import torch
from torch.utils import data as D
import os
from datetime import datetime
from . import preprocessing
import pandas as pd
from skimage import io
import cv2

"""
Functionalities for reading data for training
"""


class FundusDataset(D.Dataset):
    """Fundus Dataset loader"""

    def __init__(self, root_dir=None, csv_name=None,
                 transform_dic=None, flip_labels=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform_dic (dictionary): Optional transform to be applied
                on a sample.
        """
        self.transform_dic = transform_dic
        self.labels = pd.read_csv(root_dir + csv_name)
        self.root_dir = root_dir

        self.flip_labels = flip_labels
        if transform_dic is None or transform_dic['type'] == 'default':
            transform = preprocessing.default_transform()
        elif transform_dic['type'] == 'centerCrop':
            transform = preprocessing.center_crop(transform_dic['cropSize'])
        elif transform_dic['type'] == 'resize':
            transform = preprocessing.resize(transform_dic['size'])
        elif transform_dic['type'] == 'augmented':
            transform = preprocessing.augmented()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + self.labels['image'][idx] + '.jpeg'
        img = io.imread(img_name)

        label = self.labels['level'][idx]
        if self.transform:
            img = self.transform(img)

        if self.flip_labels:
            label = (label != 1).astype(label.dtype)

        return img, torch.tensor(label).item()


def get_fundus_train(root_dir=None, transform_dic=None, original_csv_name=None, augmented_csv_name=None,
                     augmented=False, flip_labels=False, shuffle=False,
                     batch_size=32, train_len=7000, valid_len=3000, valid_rate=None, seed=None):
    """
    Returns a tuple of dicts holding dataloarders and sizes  for the keys "train"/"val"
    """
    retina_df = pd.read_csv(root_dir + original_csv_name)
    start = datetime.now()
    if augmented:
        # Read in augmented dataset
        augmented_df = pd.read_csv(root_dir + augmented_csv_name)
        # Randomly take out samples for validation
        valid_samples = retina_df['image'].sample(n=int(len(retina_df) * valid_rate), random_state=seed)
        # single out valid and training dataset
        valid_labels_df = retina_df[retina_df['image'].isin(valid_samples)]
        train_labels_df = augmented_df[~augmented_df['original_image'].isin(valid_samples)]
        # save as .csv to fit the argument requirement of FundusDataset
        valid_labels_df.to_csv(root_dir + 'valid_samples.csv')
        train_labels_df.to_csv(root_dir + 'training_samples.csv')
        valid_dataset = FundusDataset(root_dir=root_dir, csv_name='valid_samples.csv', flip_labels=flip_labels,
                                      transform_dic=transform_dic)
        train_dataset = FundusDataset(root_dir=root_dir, csv_name='training_samples.csv', flip_labels=flip_labels,
                                      transform_dic=transform_dic)
    else:
        if seed:
            torch.manual_seed(seed)
        image_dataset = FundusDataset(root_dir=root_dir, csv_name=original_csv_name, flip_labels=flip_labels,
                                      transform_dic=transform_dic)
        if valid_rate:
            valid_len = int(valid_rate * len(image_dataset))
            train_len = len(image_dataset) - valid_len
        randperm_ind = torch.randperm(len(image_dataset))
        train_ind = randperm_ind[:train_len]
        valid_ind = randperm_ind[train_len:train_len + valid_len]
        train_labels_df = retina_df.iloc[train_ind]
        valid_labels_df = retina_df.iloc[valid_ind]
        train_dataset = D.Subset(image_dataset, train_ind)
        valid_dataset = D.Subset(image_dataset, valid_ind)

    # Get the label distribution of the training set
    train_label_distributions = torch.tensor([0] * (1 + retina_df['level'].unique().max()))
    valid_label_distributions = torch.tensor([0] * (1 + retina_df['level'].unique().max()))

    train_labels = torch.tensor(train_labels_df.groupby('level').count().index).flatten()
    train_label_distributions[train_labels] = torch.tensor(
        train_labels_df.groupby('level').count()['image'].values).flatten()
    valid_labels = torch.tensor(valid_labels_df.groupby('level').count().index).flatten()
    valid_label_distributions[valid_labels] = torch.tensor(
        valid_labels_df.groupby('level').count()['image'].values).flatten()
    print('train label distribution: {}'.format(train_label_distributions))
    print('valid label distribution: {}'.format(valid_label_distributions))
    print('proportion: {}'.format(
        train_label_distributions.type(torch.float) / valid_label_distributions.type(torch.float)))
    print("training set length: {}".format(len(train_dataset)))
    print("validation set length: {}".format(len(valid_dataset)))
    dataloaders = {'train': D.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
                   'valid': D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)}
    dataset_sizes = {'train': len(train_dataset), 'valid': len(valid_dataset)}
    print("Loaded dataset in ", datetime.now() - start)
    return dataloaders, dataset_sizes, train_label_distributions, valid_label_distributions


def get_fundus_test(root_dir, transform_dic, csv_name,
                    batch_size=32, test_len=5000, test_rate=None, seed=None):
    if seed:
        torch.manual_seed(seed)
    start = datetime.now()
    test_dataset = FundusDataset(root_dir=root_dir, csv_name=csv_name, transform_dic=transform_dic)
    if test_len >= len(test_dataset):
        return D.DataLoader(test_dataset, batch_size=batch_size)
    if test_rate:
        test_len = int(test_rate * len(test_dataset))
    test, _ = D.random_split(test_dataset, [test_len, len(test_dataset) - test_len])
    dataloader = D.DataLoader(test, batch_size=batch_size)
    print("Loaded dataset in ", datetime.now() - start)
    return dataloader, test_len
