import os

import torchvision
import torch
from torch.utils import data
from PIL import Image
from cv2 import cv2
import numpy as np


class TongRen(torch.utils.data.Dataset):
    def __init__(self, train):
        super().__init__()
        self.train = train
        if train:
            self.base = r"E:\tongren\train"
            # self.base = r"/home/rinne/tongren_data/train"
        else:
            self.base = r"E:\tongren\test"
            # self.base = r"/home/rinne/tongren_data/test"
        self.dirs = sorted(os.listdir(self.base))
        # ['amb', 'fungus', 'micro', 'virus']
        self.F = []
        self.lens = []
        for i in self.dirs:
            v = os.listdir(os.path.join(self.base, i))
            self.lens.append(len(v))
            self.F.append(v)
        self.len = sum(self.lens)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        for i in range(4):
            if idx < self.lens[i]:
                p = os.path.join(self.base, self.dirs[i], self.F[i][idx])
                return self.make_item(p, i)
            else:
                idx -= self.lens[i]

    def make_item(self, img_path, img_type):
        if self.train:
            img: Image.Image = Image.open(img_path).convert("RGB")
            img = torchvision.transforms.Resize(size=[512, 512])(img)
            img = torchvision.transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
            img = torchvision.transforms.RandomRotation(degrees=[-10, 10])(img)
            img = torchvision.transforms.RandomVerticalFlip()(img)
            img = torchvision.transforms.RandomHorizontalFlip()(img)
            img: np.ndarray = np.array(img)
            gt = np.zeros([4])
            gt[img_type] = 1
            img: torch.Tensor = torchvision.transforms.ToTensor()(img)
            return img, torch.tensor(gt)
        else:
            img: Image.Image = Image.open(img_path).convert("RGB")
            img = torchvision.transforms.Resize(size=[512, 512])(img)
            ret = []
            for i in range(5):
                img2 = img.copy()
                img2 = torchvision.transforms.RandomRotation(degrees=[-10, 10])(img2)
                img2 = torchvision.transforms.RandomVerticalFlip()(img2)
                img2 = torchvision.transforms.RandomHorizontalFlip()(img2)
                img2: torch.Tensor = torchvision.transforms.ToTensor()(img2)
                ret.append(img2)
            return ret, img_type


if __name__ == '__main__':
    x = TongRen(True)
    y = x.__getitem__(0)
    cv2.imshow("", y)
    cv2.waitKey()
