import pandas as pd
from PIL import Image
from torchvision.transforms import ColorJitter, RandomAffine, Compose, functional
import os
import time
from os import listdir
from os.path import isfile, join
import re

"""
Functionalities for dataset balancing.
"""


"""--------------------------------------------Simple oversampling-------------------------------------------------"""


def generate_balanced_dataset(csv_dir='fundus/data/train/trainLabels.csv', path='fundus/data/train/trainLabels.csv',
                              seed=19960602):
    """
    Generate balanced dataset via oversampling
    Distribution of the levels in the fundus training dataset:
    label count
    0	  25808
    1	  2443
    2	  5291
    3	  873
    4	  708
    So to construct a balanced dataset, we reduce the number of samples with label 0 and sample multiple times the images
    with level 3 and 4.
    :param csv_dir: path to the original dataset
    :param path: path to store the generated dataset
    :param seed: random seed to assure reproducbility
    """
    df = pd.read_csv(csv_dir)
    random_level0 = df[df['level'] == 0].sample(n=5000, random_state=seed)
    df = df[df['level'] != 0]
    df = df.append(random_level0)
    df = df.append(df[df['level'] == 1])
    df = df.append(df[(df['level'] == 3) | (df['level'] == 4)].sample(n=2000, random_state=seed))
    df.to_csv(index=False, path_or_buf=path + '_' + seed)


"""----------------------Add augmented images to make the fundus dataset balanced-------------------"""


def random_number(alpha, beta):
    return beta * np.random.rand() + alpha * np.random.rand()


def augment_images(file_path, lst_imgs, index):
    """
    Randomly Rotates, translate, shear and change brightness and contrast of samples in dataset
    INPUT
        file_path: file path to save the images.
        lst_imgs: list of image strings.
    OUTPUT
        Augmented images saved at file_path
    """

    for l in lst_imgs:
        img = Image.open(file_path + str(l) + '.jpeg')
        tform = Compose([RandomAffine(degrees=(-30, 10), translate=(0, 0.05), scale=(0.9, 1.1), shear=10,
                                      fillcolor=(128, 128, 128)), ColorJitter(brightness=0.15, contrast=0.15)])
        img = tform(img)
        try:
            img.save(file_path + str(l) + '_augment_' + str(index) + '.jpeg', "JPEG")
        except:
            print(file_path + str(l) + '.jpeg')


def mirror_images(file_path, lst_imgs):
    for l in lst_imgs:
        img = Image.open(file_path + str(l) + '.jpeg')
        img = functional.hflip(img)
        try:
            img.save(file_path + str(l) + '_flip_' + str(index) + '.jpeg', "JPEG")
        except:
            print(file_path + str(l) + '.jpeg')


def take_image_name(line):
    return line[0] + '_' + line[1]


def delete_jpeg(line):
    return '_'.join(line[:-1])


def generate_csv_file(dataset_dir):
    """
    Generate .csv files for labels of augmented images
    :param dataset_dir: directory of augmented dataset
    :return: .csv files for labels saved at dataset_dir
    """
    onlyfiles = [f for f in listdir(dataset_dir[:-1]) if isfile(join(dataset_dir[:-1], f))]
    onlyfiles.remove("trainLabels.csv")
    files = [re.split("[_.]", name) for name in onlyfiles]
    retina_df = pd.read_csv(os.path.join(dataset_dir[:-1], 'trainLabels.csv'))
    d = {'full_name': files}
    full_name_df = pd.DataFrame(d)
    full_name_df['image'] = full_name_df['full_name'].apply(take_image_name)
    full_name_df['full_name'] = full_name_df['full_name'].apply(delete_jpeg)
    df = pd.merge(full_name_df, retina_df)
    df = df.rename(columns={"full_name": "image", "image": "original_image"})
    df.to_csv(dataset_dir+"trainLabels_augmented.csv")


if __name__ == '__main__':
    start_time = time.time()
    root_dir = 'fundus_preprocessed_299/train/'
    trainLabels = pd.read_csv(root_dir + "trainLabels.csv")

    trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
    trainLabels_0 = trainLabels[trainLabels['level'] == 0]
    trainLabels_1 = trainLabels[trainLabels['level'] == 1]
    trainLabels_2 = trainLabels[trainLabels['level'] == 2]
    trainLabels_3 = trainLabels[trainLabels['level'] == 3]
    trainLabels_4 = trainLabels[trainLabels['level'] == 4]

    label_dict = {0: [i for i in trainLabels_0['image']], 1: [i for i in trainLabels_1['image']],
                  2: [i for i in trainLabels_2['image']], 3: [i for i in trainLabels_3['image']],
                  4: [i for i in trainLabels_4['image']]}

    # Rotate all images that have any level of DR
    print("dealing 4:")
    index = 0
    mirror_images(root_dir, label_dict[4])
    for _ in range(35):
        index += 1
        augment_images(root_dir, label_dict[4], index)

    print("processing 3:")
    index = 0
    mirror_images(root_dir, label_dict[3])
    for _ in range(28):
        index += 1
        augment_images(root_dir, label_dict[3], index)

    print("processing 2:")
    index = 0
    mirror_images(root_dir, label_dict[2])
    for _ in range(3):
        index += 1
        augment_images(root_dir, label_dict[2], index)

    print("processing 1:")
    index = 0
    mirror_images(root_dir, label_dict[1])
    for _ in range(8):
        index += 1
        augment_images(root_dir, label_dict[1], index)

    try:
        generate_csv_file(root_dir + 'trainLabels_augmented.csv')
    except:
        print("csv file generation failed!")

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))
