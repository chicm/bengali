import random
import cv2
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from imgaug import augmenters as iaa

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def crop_image_from_gray(img,tol=10):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

PI = np.pi
def do_random_crop_rotate_rescale(
    image,
    mode={'rotate': 10,'scale': 0.1,'shift': 0.1}
):

    dangle = 0
    dscale_x, dscale_y = 0,0
    dshift_x, dshift_y = 0,0

    for k,v in mode.items():
        if   'rotate'== k:
            dangle = np.random.uniform(-v, v)
        elif 'scale' == k:
            dscale_x, dscale_y = np.random.uniform(-1, 1, 2)*v
        elif 'shift' == k:
            dshift_x, dshift_y = np.random.uniform(-1, 1, 2)*v
        else:
            raise NotImplementedError

    #----

    height, width = image.shape[:2]

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift_x*width, dshift_y*height

    src = np.array([[-width/2,-height/2],[ width/2,-height/2],[ width/2, height/2],[-width/2, height/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+width/2 +tx
    y = (src*[sin, cos]).sum(1)+height/2+ty
    src = np.column_stack([x,y])

    dst = np.array([[0,0],[width,0],[width,height],[0,height]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)
    image = cv2.warpPerspective( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1))

    return image

def do_grid_distortion(image, distort=0.25, num_step = 10):

    # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    distort_x = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]
    distort_y = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]

    #---
    height, width = image.shape[:2]
    xx = np.zeros(width, np.float32)
    step_x = width // num_step

    prev = 0
    for i, x in enumerate(range(0, width, step_x)):
        if i <= num_step:
            start = x
            end   = x + step_x
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + step_x * distort_x[i]
            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur


    yy = np.zeros(height, np.float32)
    step_y = height // num_step

    prev = 0
    for idx, y in enumerate(range(0, height, step_y)):
        if idx <= num_step:
            start = y
            end = y + step_y
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + step_y * distort_y[idx]

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1,1,1))

    return image

class GraphemeDataset(Dataset):
    def __init__(self, df, mode, img_path, image_size=(137,236), fold=0):
        super().__init__()
        self.df = df
        self.image_size = image_size
        self.mode = mode
        self.fold = fold
        self.img_path = img_path
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.train_df = self.df[['image_id', 'grapheme_root', \
            'vowel_diacritic', 'consonant_diacritic']]\
            [self.df['fold'] != self.fold].values
            print(self.train_df.shape[0])
            self.num_data = self.train_df.shape[0]
            print('set dataset mode: train')

        elif self.mode == 'val':
            self.val_df = self.df[['image_id', 'grapheme_root', \
            'vowel_diacritic', 'consonant_diacritic']]\
            [self.df['fold'] == self.fold].values
            print(self.val_df.shape[0])
            self.num_data = self.val_df.shape[0]
            print('set dataset mode: train')

        print('data num: ' + str(self.num_data))

    def __len__(self):
        return self.num_data

    def __getitem__(self,idx: int):
        if self.mode == 'train':
            #sequence: root, vowel, consonant
            img_id, lb1, lb2, lb3 = self.train_df[idx, :]
            image = np.load(os.path.join(self.img_path, img_id + '.npy'))
            image = 255-image
            if self.image_size[1] != 236 and self.image_size[0]!= 137:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            #(h,w) to (h,w,1)
            image = np.expand_dims(image, axis = -1)
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image / 255.0
        elif self.mode == 'val':
            #sequence: root, vowel, consonant
            img_id, lb1, lb2, lb3 = self.val_df[idx, :]
            image = np.load(os.path.join(self.img_path, img_id + '.npy'))
            image = 255-image
            if self.image_size[1] != 236 and self.image_size[0]!= 137:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            image = np.expand_dims(image, axis = -1)
            
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image / 255.0
        return torch.FloatTensor(image), lb1, lb2, lb3


class GraphemeDataset_aux(Dataset):
    def __init__(self, df, mode, img_df, image_size=(137,236), fold=3):
        super().__init__()
        self.df = df
        self.img_df = img_df
        self.image_size = image_size
        self.mode = mode
        self.fold = fold
        #self.img_path = img_path
        self.set_mode(mode)

        
    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.train_df = self.df[['image_id', 'grapheme_root', \
            'vowel_diacritic', 'consonant_diacritic', 'word_label']]\
            [self.df['fold'] != self.fold].values
            print(self.train_df.shape[0])
            self.num_data = self.train_df.shape[0]
            print('set dataset mode: train')

        elif self.mode == 'val':
            self.val_df = self.df[['image_id', 'grapheme_root', \
            'vowel_diacritic', 'consonant_diacritic', 'word_label']]\
            [self.df['fold'] == self.fold].values
            print(self.val_df.shape[0])
            self.num_data = self.val_df.shape[0]
            print('set dataset mode: train')

        print('data num: ' + str(self.num_data))

    def __len__(self):
        return self.num_data
    
    def get_img(self, img_id):
        return 255 - self.img_df.loc[img_id].values.reshape(137, 236).astype(np.uint8)

    def __getitem__(self,idx: int):
        if self.mode == 'train':
            #sequence: root, vowel, consonant
            img_id, lb1, lb2, lb3, lb4 = self.train_df[idx, :]
            #image = np.load(os.path.join(self.img_path, img_id + '.npy'))
            #image = 255-image
            image = self.get_img(img_id)

            if self.image_size[1] != 236 and self.image_size[0]!= 137:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            #(h,w) to (h,w,1)
            image = np.expand_dims(image, axis = -1)
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image / 255.0
        elif self.mode == 'val':
            #sequence: root, vowel, consonant
            img_id, lb1, lb2, lb3, lb4 = self.val_df[idx, :]
            #image = np.load(os.path.join(self.img_path, img_id + '.npy'))
            #image = 255-image
            image = self.get_img(img_id)

            if self.image_size[1] != 236 and self.image_size[0]!= 137:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            image = np.expand_dims(image, axis = -1)
            
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image / 255.0
        return torch.FloatTensor(image), lb1, lb2, lb3, lb4
