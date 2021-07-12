from torchvision.models import densenet169, shufflenet_v2_x2_0, shufflenet_v2_x1_5
from torch.optim import ASGD
import torch.nn as nn
import sys
from dataset import build_loader
import os
from torchtoolbox.tools import mixup_data, mixup_criterion
import argparse
import time
import glob


sys.path.append('utils')
from radam import RAdam
from cyclicLR import CyclicCosAnnealingLR, LearningRateWarmUP
from losses import HybridCappaLoss
from fmix import *
from cutmix import *



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=800, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()


def get_net():
    model = shufflenet_v2_x2_0(pretrained=False)
    model.fc = nn.Sequential(
                torch.nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            )
    # input = torch.randn(2, 3, 256, 256)
    # output = model(input)
    # print(output.size())
    return model.to(DEVICE)



class Main(object):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = 2
        # create model
        self.model = get_net()

        # 超参数设置
        # self.criteration = LSRCrossEntropyLossV2(lb_smooth=0.2, lb_ignore=255)
        self.criteration = HybridCappaLoss()
        self.optimizer = ASGD(params=self.model.parameters(), lr=0.001, weight_decay=0.0001)
        milestones = [5 + x * 200 for x in range(5)]
        print(f'milestones:{milestones}')
        scheduler_c = CyclicCosAnnealingLR(self.optimizer, milestones=milestones, eta_min=5e-5)
        # # scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=4, verbose=True)
        self.scheduler = LearningRateWarmUP(optimizer=self.optimizer, target_iteration=5, target_lr=0.003,
                                       after_scheduler=scheduler_c)
        self.mix_up = False
        if self.mix_up:
            print("using mix_up")
        self.cutMix = False
        if self.cutMix:
            print("using cutMix")
        self.fmix = False
        if self.fmix:
            print("using fmix")

    def train_one_epoch(self, train_loader, val_loader):
        self.model.train()
        train_loss_sum, train_acc_sum = 0.0, 0.0
        for img, label in train_loader:
            # if len(label) <= 1:
            #     continue
            img, label = img.to(DEVICE), label.to(DEVICE)
            width, height = img.size(-1), img.size(-2)
            self.optimizer.zero_grad()
            if self.mix_up:
                img, labels_a, labels_b, lam = mixup_data(img, label, alpha=0.2)
                output = self.model(img)
                loss = mixup_criterion(self.criteration, output, labels_a, labels_b, lam)
            elif self.cutMix:
                img, targets = cutmix(img, label)
                target_a, target_b, lam = targets
                output = self.model(img)
                loss = self.criteration(output, target_a) * lam + self.criteration(output, target_b) * (1. - lam)
            elif self.fmix:
                data, target = fmix(img, label, alpha=1., decay_power=3., shape=(width, height))
                targets, shuffled_targets, lam = target
                output = self.model(data)
                loss = self.criteration(output, targets) * lam + self.criteration(output, shuffled_targets) * (1 - lam)
            else:
                output = self.model(img)
                loss = self.criteration(output, label)
            loss.backward()
            _, preds = torch.max(output.data, 1)
            correct = (preds == label).sum().item()
            train_acc_sum += correct

            train_loss_sum += loss.item()
            self.optimizer.step()

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_acc = train_acc_sum / len(train_loader.dataset)

        val_acc_sum = 0.0
        valid_loss_sum = 0
        self.model.eval()
        for val_img, val_label in val_loader:
            # if len(val_label) <= 1:
            #     continue
            val_img, val_label = val_img.to(DEVICE), val_label.to(DEVICE)
            val_output = self.model(val_img)
            _, preds = torch.max(val_output.data, 1)
            correct = (preds == val_label).sum().item()
            val_acc_sum += correct

            loss = self.criteration(val_output, val_label)
            valid_loss_sum += loss.item()

        val_acc = val_acc_sum / len(val_loader.dataset)
        val_loss = valid_loss_sum / len(val_loader.dataset)
        return train_loss, train_acc, val_loss, val_acc

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''

        train_loader, val_loader = build_loader(self.cfg)
        max_correct = 0
        for epoch in range(args.EPOCHS):
            self.scheduler.step(epoch)
            train_loss, train_acc, val_loss, val_acc = self.train_one_epoch(train_loader, val_loader)
            start = time.strftime("%H:%M:%S")
            print(
                  f"epoch:{epoch + 1}/{args.EPOCHS} | ⏰: {start}   ",
                  f"Training Loss: {train_loss:.6f}.. ",
                  f"Training Acc:  {train_acc:.6f}.. ",
                  f"validation Acc: {val_acc:.6f}.. "
                  )


            if val_acc > max_correct:
                max_correct = val_acc
                torch.save(self.model, cfg.save_path + '/' + f"model_{val_acc:.3f}.pth")
                # torch.save(self.model, MODEL_PATH + '/' + "best.pth")
                print('find optimal model')

            for path in sorted(glob.glob(f'{self.cfg.save_path}/*.pth'))[:-3]:
                os.remove(path)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    from Config import Config
    cfg = Config()
    mkdir(cfg.save_path)
    main = Main(cfg)
    main.train()