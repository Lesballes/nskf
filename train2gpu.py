#encoding: utf-8
import torch
import torch.nn.functional as F
from dgmr import (
    DGMR,
    Generator,
    Discriminator,
    TemporalDiscriminator,
    SpatialDiscriminator,
    Sampler,
    LatentConditioningStack,
    ContextConditioningStack,
)

from pytorch_lightning import Trainer
from glob import glob
import argparse
from PIL import Image
import numpy as np
import torch.utils.data
import os

# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
#-----------------------------------------------#
os.environ['CUDA_VISIBLE_DEVICES'] ='6,7'
forecast_steps = 8
gpunum = 2
stra = "ddp"
# num = 2
#-----------------------------------------------#
class DS(torch.utils.data.Dataset):
    def __init__(self, bs=2):
        #借助parser获得图片
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_dir', type=str, default='./resizedata')
        args = parser.parse_args()
        self.img_paths = glob('{:s}/*'.format(args.img_dir))
        #num就是一口喂入网络的图片数量
        num = int(len(self.img_paths)/(forecast_steps+4))
        #整合batch
        truepre = np.zeros((num, forecast_steps+4, 1, 256, 256))
        img1 = np.zeros((1, 256, 256), dtype=float)
        for i in range(num):
            for j in range(forecast_steps+4):
                img = Image.open(self.img_paths[j])
                img = np.array(img, dtype=np.uint8)  # 路径转换为图片  #x = torch.tensor(x)
                img1[0, :, :] = img
                truepre[i, j, :, :, :] = img1
        #转为tensor
        self.ds = torch.tensor(truepre)
        self.ds = self.ds.to(torch.float32)

        # self.ds = torch.rand((bs, forecast_steps + 4, 1, 256, 256))  # forecast_steps+4中的4是真实值，fore+4是真实值加预测值。
        # #ds.shape = [6,18,1,256,256]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
         # return img, self.img_paths[item] #数据、路径
        return (self.ds[item, 0:4, :, :], self.ds[item, 4:4+forecast_steps, :, :])

train_loader = torch.utils.data.DataLoader(DS(),batch_size=1)
val_loader = torch.utils.data.DataLoader(DS(),batch_size=1)

torch.cuda.empty_cache()
trainer = Trainer(gpus = gpunum, max_epochs=5, strategy=stra)
model = DGMR(forecast_steps=forecast_steps)

trainer.fit(model, train_loader,val_loader)

