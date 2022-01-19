import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import xml.etree.cElementTree as et
def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y
def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
def read_label_xml(path_label):
    tree=et.parse(path_label)
    root=tree.getroot()
    obj_num = 0
    for Object in root.findall('object'):
        name=Object.find('name').text
        obj_num += 1
    return obj_num
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path,voc_names, batch_size, img_size=416, augment=True, multi_scale=False):
        print('LoadImagesAndLabels init : ',path)
        with open(voc_names, 'r') as f:
            label_map = f.readlines()
        label_voc_dict = {}
        obj_num_sum = 0
        for i in range(len(label_map)):
            label_map[i] = label_map[i].strip()
            print(i,') ',label_map[i])
            label_voc_dict[label_map[i]] = i
        print("label_voc_dict : {}".format(label_voc_dict))
        img_files = []
        label_files = []
        s_idx = 0
        for file in os.listdir(path):
            if ".jpg" in file:
                path_img = path + file
                path_label = path_img.replace(".jpg",".xml")
                if not os.access(path_label,os.F_OK):
                    continue
                obj_num = read_label_xml(path_label)
                if obj_num == 0 :
                    continue
                obj_num_sum += obj_num
                img_files.append(path_img)
                label_files.append(path_label)
                s_idx += 1
                print("  Init LoadImagesAndLabels <{:6d}> - images   ".format(s_idx),end = "\r")
        print()
        self.label_voc_dict = label_voc_dict
        self.img_files = img_files
        assert len(self.img_files) > 0, 'No images found in %s' % path
        self.img_size = img_size
        self.batch_size = batch_size
        self.multi_scale = multi_scale
        self.augment = augment
        self.scale_index = 0
        if self.multi_scale:
            self.img_size = img_size  
            print("Multi scale images training, init img_size", self.img_size)
        else:
            print("Fixed scale images, img_size", self.img_size)
        self.label_files = label_files
        print("init voc data_iter done ~")
        print("obj_num_sum : {}".format(obj_num_sum))
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        if self.multi_scale and (self.scale_index % self.batch_size == 0)and self.scale_index != 0:
            self.img_size = random.choice(range(12, 15)) * 32
        if self.multi_scale:
            self.scale_index += 1
            if self.scale_index >= (100*self.batch_size):
                self.scale_index = 0
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        img = cv2.imread(img_path)  
        assert img is not None, 'File Not Found ' + img_path
        augment_hsv = random.random() < 0.5 
        if self.augment and augment_hsv:
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = (random.random() * 2 - 1) * fraction + 1  # a in [-0,5, 1.5]
            S *= a
            if a > 1:
                np.clip(S, None, 255, out=S)
            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, None, 255, out=V)
            img_hsv[:, :, 1] = S  # .astype(np.uint8)
            img_hsv[:, :, 2] = V  # .astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.img_size, augment=self.augment)
        tree=et.parse(label_path)
        root=tree.getroot()
        labels = []
        x = []
        for Object in root.findall('object'):
            name_=Object.find('name').text
            bndbox=Object.find('bndbox')
            xmin= np.float32((bndbox.find('xmin').text))
            ymin= np.float32((bndbox.find('ymin').text))
            xmax= np.float32((bndbox.find('xmax').text))
            ymax= np.float32((bndbox.find('ymax').text))
            xmin = np.clip(xmin,0,w-1)
            ymin = np.clip(ymin,0,h-1)
            xmax = np.clip(xmax,0,w-1)
            ymax = np.clip(ymax,0,h-1)
            x_mid = (xmax + xmin)/2./float(w)
            y_mid = (ymax + ymin)/2./float(h)
            w_box = (xmax-xmin)/float(w)
            h_box = (ymax-ymin)/float(h)
            x.append((self.label_voc_dict[name_],x_mid,y_mid,w_box,h_box))
        x = np.array(x, dtype=np.float32)
        if x.size > 0:
            labels = x.copy()
            labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-30, 30), translate=(0.10, 0.10), scale=(0.9, 1.1))
        nL = len(labels)  
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size
        if self.augment:
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
            ud_flip = True
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img, dtype=np.float32) 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return torch.from_numpy(img), labels_out, img_path, (h, w)
    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, hw
def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    shape = img.shape[:2] 
    ratio = float(height) / max(shape)  
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  
    dh = (height - new_shape[1]) / 2  
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh
def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2
    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)
    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)
    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                borderValue=borderValue) 
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targetsa
