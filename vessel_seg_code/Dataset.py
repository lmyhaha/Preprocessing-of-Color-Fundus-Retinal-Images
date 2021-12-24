import os
import cv2
import numpy as np

import torchvision
import random
from torchvision import transforms as tfs

from torch.utils.data import Dataset
from torchvision.utils import save_image

from color_normalization import *


def rand_crop(data, label, img_w, img_h):
    width1 = random.randint(0, data.shape[0] - img_w)
    height1 = random.randint(0, data.shape[1] - img_h)
    width2 = width1 + img_w
    height2 = height1 + img_h

    data = data[height1: height2, width1:width2, :]
    label = label[height1: height2, width1:width2]

    return data, label


class DRIVEDatasets(Dataset):

    def __init__(self, path, img_size=400):
        self.path = path
        self.img_size = img_size
        # 语义分割需要的图片的图片和标签
        self.images = os.listdir(os.path.join(path, "images"))
        self.targets = os.listdir(os.path.join(path, "1st_manual"))
        # v3版本参数 0.05 0.1 0.1 0.1
        # v4版本参数 0.1  0.2 0.2 0.2
        # v5版本参数 0.2  0.4 0.4 0.4
        self.trans_img = tfs.Compose([tfs.ToTensor(),
                                      tfs.ColorJitter(hue=0.05, contrast=0.1, brightness=0.1, saturation=0.1)])
        self.trans_label = tfs.Compose([tfs.ToTensor()])

    def __len__(self):
        return len(self.images)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸+
        _w = _h = self.img_size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        # 拿到的图片和标签
        name1 = self.images[index]
        name2 = self.targets[index]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]
        # 读取原始图片和标签，并转RGB
        input_img = cv2.imread(os.path.join(img_path[0], name1))
        _, target = cv2.VideoCapture(os.path.join(img_path[1], name2)).read()
        target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        input_img = self.__trans__(input_img)
        target = self.__trans__(target)

        ret, target = cv2.threshold(target, 20, 255, cv2.THRESH_BINARY)

        # input_img, target = rand_crop(input_img, target, 256, 256)

        return self.trans_img(input_img), self.trans_label(target)


class DRIVEDatasets_for_test(Dataset):

    def __init__(self, path, img_size=400):
        self.path = path
        self.img_size = img_size
        # 语义分割需要的图片的图片和标签
        self.images = os.listdir(os.path.join(path, "images"))
        self.targets = os.listdir(os.path.join(path, "1st_manual"))
        # v3版本参数 0.05 0.1 0.1 0.1
        # v4版本参数 0.1  0.2 0.2 0.2
        # v5版本参数 0.2  0.4 0.4 0.4
        self.trans_img = tfs.Compose([tfs.ToTensor()])
        self.trans_label = tfs.Compose([tfs.ToTensor()])

    def __len__(self):
        return len(self.images)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸+
        _w = _h = self.img_size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        # 拿到的图片和标签
        name1 = self.images[index]
        name2 = self.targets[index]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ("images", "1st_manual")]
        # 读取原始图片和标签，并转RGB
        input_img = cv2.imread(os.path.join(img_path[0], name1))
        _, target = cv2.VideoCapture(os.path.join(img_path[1], name2)).read()
        target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        input_img = self.__trans__(input_img)
        target = self.__trans__(target)

        ret, target = cv2.threshold(target, 20, 255, cv2.THRESH_BINARY)

        return self.trans_img(input_img), self.trans_label(target)


if __name__ == '__main__':
    i = 1
    dataset = DRIVEDatasets(r"Data_trans/training", 292)
    for a, b in dataset:
        print(i)
        print(a.shape)
        print(b.shape)
        save_image(a, f"./img/{i}.jpg", nrow=1)
        save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 5:
            break
