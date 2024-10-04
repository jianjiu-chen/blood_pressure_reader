import os

import cv2
import pandas as pd

# create directories to store cropped images
dir_1 = 'train/datasets'
dir_2 = 'BP_ROI'

dir_to_create_1 = os.path.join(dir_1, dir_2, 'train/images')
dir_to_create_2 = os.path.join(dir_1, dir_2, 'train/labels')
dir_to_create_3 = os.path.join(dir_1, dir_2, 'val/images')
dir_to_create_4 = os.path.join(dir_1, dir_2, 'val/labels')

os.makedirs(dir_to_create_1, exist_ok=True)
os.makedirs(dir_to_create_2, exist_ok=True)
os.makedirs(dir_to_create_3, exist_ok=True)
os.makedirs(dir_to_create_4, exist_ok=True)

# loop: open an image, read a txt file, crop the image, save the ROI to a folder
train_images = sorted(os.listdir('train/datasets/BP_location/train/images'))
train_labels = sorted(os.listdir('train/datasets/BP_location/train/labels'))
val_images = sorted(os.listdir('train/datasets/BP_location/val/images'))
val_labels = sorted(os.listdir('train/datasets/BP_location/val/labels'))

for train_image_i, train_label_i in zip(train_images, train_labels):
    train_image_obj_i = cv2.imread(os.path.join('train/datasets/BP_location/train/images', train_image_i))
    image_w = train_image_obj_i.shape[1]
    image_h = train_image_obj_i.shape[0]

    train_label_obj_i = pd.read_csv(os.path.join('train/datasets/BP_location/train/labels', train_label_i),
                                    delim_whitespace=True, header=None)
    x_mid_box = train_label_obj_i.iloc[0, 1] * image_w
    y_mid_box = train_label_obj_i.iloc[0, 2] * image_h
    box_w = train_label_obj_i.iloc[0, 3] * image_w
    box_h = train_label_obj_i.iloc[0, 4] * image_h

    train_image_obj_i_ROI = train_image_obj_i[int(y_mid_box - box_h / 2):int(y_mid_box + box_h / 2),
                                              int(x_mid_box - box_w / 2):int(x_mid_box + box_w / 2)]
    cv2.imwrite(os.path.join('train/datasets/BP_ROI/train/images', train_image_i), train_image_obj_i_ROI)

for val_image_i, val_label_i in zip(val_images, val_labels):
    val_image_obj_i = cv2.imread(os.path.join('train/datasets/BP_location/val/images', val_image_i))
    image_w = val_image_obj_i.shape[1]
    image_h = val_image_obj_i.shape[0]

    val_label_obj_i = pd.read_csv(os.path.join('train/datasets/BP_location/val/labels', val_label_i),
                                  delim_whitespace=True, header=None)
    x_mid_box = val_label_obj_i.iloc[0, 1] * image_w
    y_mid_box = val_label_obj_i.iloc[0, 2] * image_h
    box_w = val_label_obj_i.iloc[0, 3] * image_w
    box_h = val_label_obj_i.iloc[0, 4] * image_h

    val_image_obj_i_ROI = val_image_obj_i[int(y_mid_box - box_h / 2):int(y_mid_box + box_h / 2),
                                          int(x_mid_box - box_w / 2):int(x_mid_box + box_w / 2)]
    cv2.imwrite(os.path.join('train/datasets/BP_ROI/val/images', val_image_i), val_image_obj_i_ROI)
