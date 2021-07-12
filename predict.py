import torch
import os
import glob
import cv2 as cv
from torch.autograd import Variable
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Cutout, CoarseDropout, Normalize, ElasticTransform
)
from albumentations.pytorch.transforms import ToTensorV2, ToTensor


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Prediction(object):
    def __init__(self):
        self.transformation = Compose([
            Normalize(),
            ToTensorV2(),
        ])

    def load_model(self, modelPath):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        # pass
        # print(MODEL_PATH+'/'+'best.pth')
        self.model = torch.load(modelPath)
        self.model = self.model.to(device)

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/cloudy/00000.jpg"}
        :return: 模型预测成功中户 {"label": 0}
        '''
        print(image_path)
        # Normalize = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}
        result = []

        img = cv.imread(image_path)
        tensor = self.transformation(image=img)['image']
        tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)

        output = self.model(tensor.to(device))

        pred = output.max(1, keepdim=True)[1]

        # output = tta.fliplr_image2label(self.model, tensor.to(device))
        # pred = output.max(1, keepdim=True)[1].item()

        return {"label": pred}


if __name__ == '__main__':
    predict = Prediction()
    imglist = glob.glob('./data/neg/*.tif')
    predict.load_model('./results/model_0.289.pth')
    for imgPath in imglist:
        print(predict.predict(imgPath))