from glob import glob
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='./tdata')
args = parser.parse_args()
img_paths = glob('{:s}/*'.format(args.img_dir))

# num = 6
# # num = len(img_paths)/18
# img618 = np.zeros((num,18,1, 256, 256))
# img1= np.zeros((1, 256, 256))
# for i in range(6):
#     for j in range(18):
#         img = Image.open(img_paths[j])
#         img = np.array(img, dtype=np.uint8) #路径转换为图片  #x = torch.tensor(x)
#
#         img1[0,:,:] = img
#         img618[i,j,:,:,:] = img1
# img618 = torch.tensor(img618)
#
#     # if i  == 0:
#     #     img18 = img
#     # img18 = np.vstack((img18,img))
# # img18.reshape(18,1,256,256)
# #prepare data
# a = np.arange(16)
# a1=a.reshape([4,4])
# a2=a1
# a3=a1
# ### way 1
#
# a4=np.array([a1,a2,a3])
# # print(a4.shape)
# ### way 2
# a5=np.vstack((a1,a2,a3)).reshape(3,4,4)
# ### way 3
# a6=np.zeros((3,4,4))
# a6[0,:,:]=a1
# a6[1,:,:]=a2
# a6[2,:,:]=a3
# #
# ds = torch.rand((2, 8+ 4, 1, 256, 256))
# # print(ds.shape)
# --------------------------------#
# num = 6
# forecast_steps = 12
# parser = argparse.ArgumentParser()
# parser.add_argument('--img_dir', type=str, default='./tdata')
# args = parser.parse_args()
#
# img_paths = glob('{:s}/*'.format(args.img_dir))
#             #整合数据集的输入
#             # num = len(img_paths)/18
# img618 = np.zeros((num, forecast_steps, 1, 256, 256))
# img1 = np.zeros((1, 256, 256))
# for i in range(num):
#     for j in range(forecast_steps):
#         img = Image.open(img_paths[j])
#         img = np.array(img, dtype=np.uint8)  # 路径转换为图片  #x = torch.tensor(x)
#         img1[0, :, :] = img
#         img618[i, j, :, :, :] = img1
# ds = torch.tensor(img618)
# ds.float()
# # print(ds)

#-----------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='./tdata')
args = parser.parse_args()

img_paths = glob('{:s}/*'.format(args.img_dir))
#整合数据集的输入
# num = len(img_paths)/18
forecast_steps = 4
num = 2
truepre = np.zeros((num, forecast_steps+4, 1, 256, 256))
img1 = np.zeros((1, 256, 256), dtype=float)
for i in range(num):
    for j in range(forecast_steps+4):
        img = Image.open(img_paths[j])
        img = np.array(img, dtype=np.uint8)  # 路径转换为图片  #x = torch.tensor(x)
        img1[0, :, :] = img
        truepre[i, j, :, :, :] = img1

ds = torch.tensor(truepre)
ds = ds.to(torch.float32)
# print(ds.shape)
# print(ds)

ds1= torch.rand((2, forecast_steps + 4, 1, 256, 256))
ds2 = ds1[:,0:1,:,:]
print(ds2.shape)
# print(ds1)