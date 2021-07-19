# %%
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import h5py
# %%
DIR = '../plant-pathology-2021-fgvc8'
csv_path = f'{DIR}/train.csv'
train = pd.read_csv(csv_path)
#%%
train['labels'] = train['labels'].apply(lambda string: string.split(' '))

s = list(train['labels'])
mlb = MultiLabelBinarizer()
trainx = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=train.index)
labels = np.array(trainx)

train_data = list(trainx.sum().keys())
train_data = pd.concat([train['image'], trainx], axis=1)


#%%
albumentation_list = [A.RandomSunFlare(p=1), 
                      A.RandomFog(p=1), 
                      A.RandomBrightness(p=1), 
                      A.Rotate(p=1, limit=90),
                      A.RGBShift(p=1), 
                      A.RandomSnow(p=1),
                      A.HorizontalFlip(p=1), 
                      A.VerticalFlip(p=1), 
                      A.RandomContrast(limit = 0.5,p = 1),
                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50),
                      A.Cutout(p=1),
                      A.JpegCompression(p=1),
                      A.CoarseDropout(p=1),
                      A.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003, 12.75), per_channel=False, p=1),
                      A.IAAAffine(scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0, mode='reflect', p=1),
                      A.IAAAffine(rotate=90., p=1),
                      A.IAAAffine(rotate=180., p=1)]
# %%
img_matrix_list = []
labels_matrix_list = []
HEIGHT = 512
WIDTH = 512

for i, image in enumerate(train_data.image):

    image_path = f'{DIR}/train_images/{image}'
    chosen_image = cv2.imread(image_path)
    chosen_image = cv2.resize(chosen_image,(HEIGHT, WIDTH))
    l = labels[i]

    cv2.imwrite(f'{DIR}/aug_images/{image}',chosen_image)
    img_matrix_list.append(image)
    labels_matrix_list.append(l)

    for j, aug_type in enumerate(albumentation_list):
        img_name = f'{image}_{j}.jpg'
        img = aug_type(image = chosen_image)['image']
        img = cv2.resize(img,(HEIGHT, WIDTH))
        cv2.imwrite(f'{DIR}/aug_images/{img_name}',img)
        img_matrix_list.append(img_name)
        labels_matrix_list.append(l)

# %%
hf = h5py.File(f'{DIR}/aug_train_images2.h5', 'w')
hf.create_dataset('train_images', data=img_matrix_list)
hf.create_dataset('labels', data=labels_matrix_list)
hf.close()

# %%
