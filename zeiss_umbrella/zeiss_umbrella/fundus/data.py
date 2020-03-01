import torch
from torch.utils import data as D
import os, io
from datetime import datetime
from . import preprocessing
from zeiss_umbrella.fundus.adversarial import get_diff
from zeiss_umbrella.fundus.quality_augmentation.transform import preset_augment
from zeiss_umbrella.fundus.quality_augmentation.make_dataset import convert_to_hdf5, convert_to_hdf5_jpeg
import pandas as pd
from skimage.io import imread
import numpy as np
import cv2
import warnings
import h5py


class FundusDataset(D.Dataset):
    """Fundus Dataset loader"""

    def __init__(self, root_dir=None, csv_name=None, df=None, corruption_dic=None, phase=None,
                 transform_dic=None, flip_labels=False, gaussian_noise=None):
        """
        Three ways of loading dataset:
        1. from a .csv file
        2. from a pandas dataframe
        3. from a .h5 corruption dataset
        These ways are mutually exclusive
        root_dir: directory where images and label csv are stored
        csv_name: the name of the csv label file stored in root_dir
        df: dataframe with image names and labels
        corruption_dic: dictionary storing setting of corruption dataset
        phase: 'train' or 'valid' (specifically for .h5 files)
        transform_dic: dictionary storing setting of transformation.
                       format: {'type': 'normalize'/'default'/'augmented'}
                                'corruption': 'true' or 'false' perform corruption before above transformation or not
                                if true should define 'ops' and 'or_less' in the transform_dic
        flip_labels: flip labels or not
        gaussian_noise: amplitude of gaussian noise added after transformations defined in transform_dic are performed
        """
        self.corruption_dic = corruption_dic
        self.transform_dic = transform_dic
        self.gaussian_noise = gaussian_noise
        if self.corruption_dic:
            h5_file = h5py.File(corruption_dic['h5_path'], "r")
            if self.corruption_dic['valid_corruption']:
                self.data_x = h5_file.get(phase).get('augmented')
            else:
                if phase == 'train':
                    self.data_x = h5_file.get(phase).get('augmented')
                elif phase == 'valid':
                    self.data_x = h5_file[h5_file['valid']['original'][0]]
                else:
                    raise ValueError("unknown phase")
            self.labels = h5_file.get(phase).get('label')

        elif df.__class__.__name__ != 'NoneType':
            self.labels = df.copy()
        else:
            self.labels = pd.read_csv(os.path.join(root_dir, csv_name))
        self.root_dir = root_dir

        self.flip_labels = flip_labels
        if transform_dic['type'] == 'normalize':
            transform = preprocessing.normalize()
        elif transform_dic['type'] == 'augmented':
            transform = preprocessing.augmented()
        else:
            transform = preprocessing.default()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.corruption_dic:
            if self.corruption_dic['output_type'] == 'jpeg':
                img = imread(io.BytesIO(self.data_x[idx]))
                label = int(self.labels[idx])
            else:
                img = self.data_x[idx]
                label = int(self.labels[idx])
        else:
            img_name = os.path.join(self.root_dir, self.labels['image'][idx] + '.jpeg')
            img = imread(img_name)
            label = self.labels['level'][idx]

        if self.transform_dic.get('corruption', None):
            img = preset_augment(img, ops=self.transform_dic['ops'], or_less=self.transform_dic['or_less'])

        if self.transform:
            img = self.transform(img)

        if self.gaussian_noise:
            img += self.gaussian_noise * torch.randn_like(img)

        if self.flip_labels:
            label = (label != 1).astype(label.dtype)

        return img, torch.tensor(label).item()


def normal_balance(df, root_dir, csv_name, class_name, seed):
    """
    Generate balanced dataframe saved as .csv file from input dataframe
    :param seed: seed for reproducibility
    :param class_name: Name of the class to be balanced
    :param root_dir: path to the folder saving the resulting .csv file
    :param csv_name: name of the csv file
    :param df: input dataframe
    """
    df_copy = df.copy()
    dominant_class = df_copy.groupby(class_name).count().idxmax()[0]
    num_dominant_class = df_copy.groupby(class_name).count().max()[0]
    for label_class in df_copy[class_name].unique():
        if label_class != dominant_class:
            sample_class = df_copy[df_copy[class_name] == label_class].copy()
            for _ in range(int(num_dominant_class / len(sample_class)) - 1):
                df_copy = df_copy.append(sample_class)
            rest_length = num_dominant_class - int(num_dominant_class / len(sample_class)) * len(sample_class)
            df_copy = df_copy.append(sample_class.sample(n=rest_length, random_state=seed))
    df_copy.to_csv(os.path.join(root_dir, csv_name))
    print('Generating {}'.format(csv_name))


def generate_distance_stat(input_csv_name, root_dir, class_name, device, batch_size):
    """
    Calculate the Euclidean distance between all sample pairs and save it as a .csv file.
    :param input_csv_name: The original dataset csv
    :param root_dir: dir where samples and dataset csv are stored
    :param class_name: the name of the label (level for fundus dataset)
    :param device: the device used to calculate distance
    :param batch_size
    :return: a .csv file with head "image1 image2 dist"
    """
    d = preprocessing.default_transform()
    df = pd.read_csv(os.path.join(root_dir, input_csv_name))
    dominant_class = df.groupby(class_name).count().idxmax()[0]
    distance_dict = {'image1': [], 'image2': [], 'dist': []}
    for label in df['level'].unique():
        if label != dominant_class:
            df_label = df[df[class_name] == label]
            # stack all the samples as a tensor
            image_tensor = torch.stack(
                [d(imread(os.path.join(root_dir, img_name + '.jpeg'))) for img_name in df_label['image']])
            counter = 0
            for ind, img_name1 in enumerate(df_label['image']):
                counter += 1
                print('progress: {:.3f}'.format(counter / len(df_label)))
                # process batch by batch
                for i in range(0, len(df_label), batch_size):
                    step_size = min(batch_size, len(df_label) - i)
                    image1_tensor = image_tensor[ind].repeat(step_size, 1, 1, 1)
                    distance = (
                        get_diff(image_tensor[i:i + step_size], image1_tensor, device=device).max(dim=1)[0]).tolist()
                    distance_dict['dist'] += distance
                distance_dict['image2'] += df_label['image'].tolist()
                distance_dict['image1'] += [img_name1] * len(df_label)
    df_distance = pd.DataFrame(distance_dict)
    df_distance[df_distance['image1'] != df_distance['image2']].reset_index(drop=True).to_csv(
        os.path.join(root_dir, 'df_distance.csv'), index=False)


def smote(k=None, root_dir=None, distance_csv_name=None, df=None, output_csv_name=None, seed=None, class_name=None):
    """
    Genarate smote examples and add the corresponding filename to the sample dataframe(df)
    :param k: number of nearest neighbours
    :param root_dir: dir where the samples and labels are stored
    :param distance_csv_name: the name of file storing distance between samples
    :param df: the dataframe from which a smoted dataframe and associate samples are generated
    :param output_csv_name: the name of the output dataframe (with smote sample names added)
    :param seed: seed for reproducbility
    :param class_name: name of the class (level for fundus dataset)
    :return: a new balanced dataset
    """
    # Read in the distance csv file
    df_distance = pd.read_csv(os.path.join(root_dir, distance_csv_name)).reset_index(drop=True)
    # Select distance information related to the samples in dataframe and add level information to df_distance
    df_distance = df_distance.merge(df, left_on='image1', right_on='image').drop('image', axis=1)
    # Take out the indices of k nearest neighbour of samples in df
    closest_ind_list = [ind for _, ind in df_distance.groupby('image1')['dist'].nsmallest(k).index]
    # collect the k nearest neighbours in a column of list called "closest images"
    df_closest = df_distance.iloc[closest_ind_list].groupby('image1') \
        .agg(lambda x: x.to_list()) \
        .rename(columns={'image2': 'closest images'})
    # convert level list to a single integer ([1,1,1,1] -> 1)
    df_closest['level'] = df_closest['level'].apply(lambda x: x[0])
    # df_closest.head -> "image1" "closest images" "level" "dist"
    dominant_class = df.groupby(class_name).count().idxmax()[0]
    num_dominant_class = df.groupby(class_name).count().max()[0]
    output_df = df.copy()
    for label_class in df[class_name].unique():
        if label_class != dominant_class:
            print('Generating smote samples for label {}'.format(str(label_class)))
            file_name_list = []
            num_samples = df.groupby('level').count().iloc[label_class][0]
            if num_samples > 0:
                n = int(num_dominant_class / num_samples)
                rest = num_dominant_class - n * num_samples
            else:
                n, rest = 0, 0
            print("n: {}".format(str(n)))
            print("rest: {}".format(str(rest)))
            # We only work on the samples with label_class
            df_closest_label = df_closest[df_closest['level'] == label_class]
            for ind in range(n - 1):
                file_name_list += generate_smote_samples(df_closest_label, root_dir, ind)
            if rest > 0:
                file_name_list += generate_smote_samples(df_closest_label.sample(n=rest, random_state=seed), root_dir,
                                                         n - 1)
            output_df = output_df.append(
                pd.DataFrame({'image': file_name_list, 'level': [label_class] * len(file_name_list)})).reset_index(
                drop=True)
    output_df.to_csv(os.path.join(root_dir, output_csv_name))


def generate_smote_samples(df_closest, root_dir, ind):
    file_name_list = []
    # Loop over images
    for image1_name, image1_series in df_closest.iterrows():
        img1 = imread(os.path.join(root_dir, image1_name + '.jpeg'))
        closest_image_list = image1_series['closest images']
        image2_name = closest_image_list[torch.randperm(len(closest_image_list))[0]]
        img2 = imread(os.path.join(root_dir, image2_name + '.jpeg'))
        t = torch.rand(size=(1, 1)).item()
        file_name = '{}_smote_{}'.format(image1_name, str(ind))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(root_dir, file_name + '.jpeg'), (t * img1 + (1 - t) * img2).astype('uint8'))
        file_name_list.append(file_name)
    return file_name_list


def get_fundus_train(root_dir=None, transform_dic=None, original_csv_name=None, flip_labels=False, shuffle=False,
                     balance=None, k=None, batch_size=32, train_len=7000, valid_len=3000, valid_rate=None, seed=None,
                     gaussian_noise=None, corruption_dic=None, valid_transform_dic=None):
    """
    Returns a tuple of dicts holding dataloarders and sizes for the keys "train"/"val"
    """
    if valid_transform_dic is None:
        valid_transform_dic = {'type': 'normalize'}
    start = datetime.now()
    retina_df = pd.read_csv(os.path.join(root_dir, original_csv_name))

    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if valid_rate:
        valid_len = int(valid_rate * len(retina_df))
        train_len = len(retina_df) - valid_len

    # Randomly generate training labels samples and valid labels samples
    randperm_ind = torch.randperm(len(retina_df))
    train_ind = randperm_ind[:train_len]
    valid_ind = randperm_ind[train_len:train_len + valid_len]
    train_labels_df = retina_df.iloc[train_ind]
    valid_labels_df = retina_df.iloc[valid_ind]

    train_dataset, valid_dataset, train_labels_df = dataset_balance(balance, train_labels_df, train_ind, valid_ind,
                                                                    root_dir, seed, original_csv_name,
                                                                    transform_dic, gaussian_noise, flip_labels, k,
                                                                    corruption_dic, valid_transform_dic)

    # If we use corruption:
    if corruption_dic:
        if not os.path.exists(corruption_dic['h5_path']):
            print('Generating ' + corruption_dic['h5_path'])
            if corruption_dic['output_type'] == 'original':
                convert_to_hdf5(train_dataset, valid_dataset, corruption_dic['img_size'], corruption_dic['h5_path'],
                                corruption_dic['ops'], epochs=1, or_less=corruption_dic['or_less'])
            elif corruption_dic['output_type'] == 'jpeg':
                convert_to_hdf5_jpeg(train_dataset, valid_dataset, corruption_dic['h5_path'], corruption_dic['ops'],
                                     epochs=1,
                                     or_less=corruption_dic['or_less'])
            else:
                raise ValueError("Unsupported corruption output type")
        print('Using EXISTED ' + corruption_dic['h5_path'])
        train_dataset = FundusDataset(corruption_dic=corruption_dic,
                                      transform_dic={'type': 'normalize'}, phase='train')
        valid_dataset = FundusDataset(corruption_dic=corruption_dic,
                                      transform_dic=valid_transform_dic, phase='valid')

    # Get the label distribution of the training set
    train_label_distributions = torch.tensor([0] * (1 + retina_df['level'].unique().max()))
    valid_label_distributions = torch.tensor([0] * (1 + retina_df['level'].unique().max()))

    train_labels = torch.tensor(train_labels_df.groupby('level').count().index).flatten()
    train_label_distributions[train_labels] = torch.tensor(
        train_labels_df.groupby('level').count()['image'].values).flatten()
    valid_labels = torch.tensor(valid_labels_df.groupby('level').count().index).flatten()
    valid_label_distributions[valid_labels] = torch.tensor(
        valid_labels_df.groupby('level').count()['image'].values).flatten()
    output_info(train_label_distributions, valid_label_distributions, train_dataset, valid_dataset)
    dataloaders = {'train': D.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
                   'valid': D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)}
    datasets = {'train': train_dataset, 'valid': valid_dataset}
    dataset_sizes = {'train': len(train_dataset), 'valid': len(valid_dataset)}
    print("Loaded dataset in ", datetime.now() - start)
    return dataloaders, datasets, dataset_sizes, train_label_distributions, valid_label_distributions


def dataset_balance(balance, train_labels_df, train_ind, valid_ind, root_dir, seed, original_csv_name,
                    transform_dic, gaussian_noise, flip_labels, k, corruption_dic, valid_transform_dic):
    # The input images of corruption should be the original ones.
    if corruption_dic is not None:
        train_transform_dic, valid_transform_dic = {'type': 'default'}, {'type': 'default'}
        gaussian_noise = None
    else:
        train_transform_dic = transform_dic

    if balance == 'normal':
        normal_balance(train_labels_df, root_dir=root_dir, csv_name='trainLabels_balanced_normal.csv',
                       class_name='level', seed=seed)
        train_dataset = FundusDataset(root_dir=root_dir, csv_name='trainLabels_balanced_normal.csv',
                                      transform_dic=train_transform_dic, gaussian_noise=gaussian_noise)
        image_dataset_valid = FundusDataset(root_dir=root_dir, csv_name=original_csv_name, flip_labels=flip_labels,
                                            transform_dic=valid_transform_dic)
        valid_dataset = D.Subset(image_dataset_valid, valid_ind)
        train_labels_df = pd.read_csv(os.path.join(root_dir, 'trainLabels_balanced_normal.csv'))

    elif balance == 'smote':
        csv_name = 'trainLabels_balanced_smote.csv'
        smote(k=k, root_dir=root_dir, distance_csv_name='df_distance.csv', df=train_labels_df,
              output_csv_name=csv_name, seed=seed, class_name='level')
        train_dataset = FundusDataset(root_dir=root_dir, csv_name=csv_name,
                                      flip_labels=flip_labels, transform_dic=train_transform_dic)
        image_dataset_valid = FundusDataset(root_dir=root_dir, csv_name=original_csv_name, flip_labels=flip_labels,
                                            transform_dic=valid_transform_dic)
        valid_dataset = D.Subset(image_dataset_valid, valid_ind)
        train_labels_df = pd.read_csv(os.path.join(root_dir, csv_name))

    else:
        image_dataset_train = FundusDataset(root_dir=root_dir, csv_name=original_csv_name, flip_labels=flip_labels,
                                            transform_dic=train_transform_dic)
        image_dataset_valid = FundusDataset(root_dir=root_dir, csv_name=original_csv_name, flip_labels=flip_labels,
                                            transform_dic=valid_transform_dic)
        train_dataset = D.Subset(image_dataset_train, train_ind)
        valid_dataset = D.Subset(image_dataset_valid, valid_ind)
    return train_dataset, valid_dataset, train_labels_df


def output_info(train_label_dist, valid_label_dist, train_dataset, valid_dataset):
    print('train label distribution: {}'.format(train_label_dist))
    print('valid label distribution: {}'.format(valid_label_dist))
    print('proportion: {}'.format(
        train_label_dist.type(torch.float) / valid_label_dist.type(torch.float)))
    print("training set length: {}".format(len(train_dataset)))
    print("validation set length: {}".format(len(valid_dataset)))


def get_fundus_test(root_dir, transform_dic, csv_name,
                    batch_size=32, test_len=5000, test_rate=None, seed=None):
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
