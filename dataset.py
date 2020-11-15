import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
# from scipy.misc import imread
from imageio import imread
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True, mask_reverse = False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)

        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        img = imread(self.data[index])
        if self.training:
            img = self.resize(img)
        else:
            img = self.resize(img, True, True, True)
        # load mask
        mask = self.load_mask(img, index)
        # augment data
        if self.training:
            if self.augment and np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]
            if self.augment and np.random.binomial(1, 0.5) > 0:
                mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(mask)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        
        #external mask, random order
        if self.mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
            mask = self.resize(mask, False)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255
        #generate random mask
        if self.mask_type == 1:
            mask = 1 - generate_stroke_mask([self.target_size, self.target_size])
            mask = (mask>0).astype(np.uint8)* 255
            mask = self.resize(mask,False)
            return mask
        
        #external mask, fixed order
        if self.mask_type == 2:
            mask_index = index
            mask = imread(self.mask_data[mask_index])
            mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
            mask = self.resize(mask, False)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

    def resize(self, img, aspect_ratio_kept = True, fixed_size = False, centerCrop=False):
        
        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
        # img = scipy.misc.imresize(img, [self.target_size, self.target_size])
        img = np.array(Image.fromarray(img).resize(size=(self.target_size, self.target_size)))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_list(self, path):
        if isinstance(path, str):
            if path[-3:] == "txt":
                line = open(path,"r")
                lines = line.readlines()
                file_names = []
                for line in lines:
                    file_names.append("../../Dataset/Places2/train/data_256"+line.split(" ")[0])
                return file_names
            if os.path.isdir(path):
                path = list(glob.glob(path + '/*.jpg')) + list(glob.glob(path + '/*.png'))
                path.sort()
                return path
            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []

def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask

def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask