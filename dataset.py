import os
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import cv2 as cv
import glob
from random import sample, shuffle

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Cutout, CoarseDropout, Normalize, ElasticTransform
)
from albumentations.pytorch.transforms import ToTensorV2, ToTensor


class ImageData(Dataset):
    def __init__(self, root, mode='train'):
        self.imglist = glob.glob(f'{root}/*/*.tif')
        self.mode = mode
        self.classDict = {'neg': 0,
                          'pos': 1}
        self.train_transformation = Compose([
            RandomRotate90(p=0.6),
            # GridDistortion(p=0.6),
            HorizontalFlip(p=0.6),
            # ElasticTransform(alpha=1, sigma=25, alpha_affine=50, p=0.75),
            # OneOf([
            #     IAAAdditiveGaussianNoise(),
            #     GaussNoise(),
            # ], p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=15, p=0.75),
            Normalize(),
            ToTensorV2(),
        ])
        self.valid_transformation = Compose([
            Normalize(),
            ToTensorV2(),
        ])

    def __getitem__(self, item):
        img = cv.imread(self.imglist[item])
        className = os.path.split(self.imglist[item])[-2].split('\\')[-1]
        label = self.classDict[className]

        if self.mode == "train":
            img = self.train_transformation(image=img)['image']
        else:
            img = self.valid_transformation(image=img)['image']
        return img, label

    def __len__(self):
        return len(self.imglist)


def build_loader(cfg):
    # Get correct indices
    num_train = len(glob.glob(f'{cfg.trainData}/*/*.tif'))
    indices = list(range(num_train))
    indices = sample(indices, len(indices))
    split = int(np.floor(0.3 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    print(f'the length of train dataset is {len(train_idx)} \n',
          f'the length of valid dataset is {len(valid_idx)} \n')
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # set sup datasets
    train_dataset = ImageData(cfg.trainData, mode='train')
    val_dataset = ImageData(cfg.trainData, mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=valid_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    return train_loader, valid_loader


if __name__=='__main__':
    from Config import Config
    cfg = Config()
    train_loader, valid_loader = build_loader(cfg)
    for img, label in train_loader:
        print('..')