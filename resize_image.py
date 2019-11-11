import torch
import os
import pandas as pd
import cv2
import numpy

"""
Resize and preprocessed images according to Kaggle Diabetic Retinopathy Competition Report of Ben Graham.
"""


def scaleRadius(img, scale):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def image_preprocessing(img, scale=500, size=(512, 512)):
    img = scaleRadius(img, scale)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30.), -4, 128)
    img2 = numpy.zeros(img.shape)
    img2 = cv2.circle(img2, center=(int(img.shape[1] / 2), int(img.shape[0] / 2)), radius=int(scale * 0.9),
                      color=(1, 1, 1), thickness=-1, lineType=8, shift=0)
    img = img * img2 + 128 * (1 - img2)

    s = size[0] / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=s, fy=s)
    img = img[:, int(img.shape[1] / 2) - int(size[0] / 2):int(img.shape[1] / 2) + int(size[0] / 2)]
    return img


def preprocess_and_save_images(root_dir="original_images/train/",
                               output_dir='fundus_resize_512/train/',
                               csv_name='trainLabels.csv', size=(256, 256)):
    """
    resize image from root_dir and save them to output_dir, assuming that label document and images are stored in the
    same directory.
    :param size: tuple, size of the output
    :param root_dir: path to the original images
    :param output_dir: path to output images
    :param csv_name: name of the label file
    """
    df = pd.read_csv(root_dir + csv_name)
    for ind, image_name in enumerate(df['image']):
        if ind % 1000 == 0:
            print('finish {} photos'.format(ind))
        read_in_path = root_dir + image_name + '.jpeg'
        save_path = output_dir + image_name + '.jpeg'
        img = cv2.imread(read_in_path)
        if img is None:
            raise FileNotFoundError
        try:
            img = image_preprocessing(img, size=size)
            cv2.imwrite(save_path, img)
            imgg = cv2.imread(save_path)
            if imgg.shape[0] != size[0]:
                print("read in shape not valid!!!!!")
                print(save_path)
                print(read_in_path)
        except:
            print(save_path)
            print(read_in_path)


preprocess_and_save_images(output_dir='fundus_preprocessed_299/train/', size=(299, 299))
