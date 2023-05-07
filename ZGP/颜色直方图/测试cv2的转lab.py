import torchvision.transforms as transforms
import torch
from scipy import stats
import cv2                      #导入 Opencv
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import color  # require skimage

# a = torch.randint(size=(1, 3, 6, 6), high=10)
# b = a[:, 1, :, :]
# print(a, b)

img_file = r"H:\Datasets\VOC2012\JPEGImages\2007_000042.jpg"#读取图片
img = cv2.imread(img_file, 1)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) #z BGR转RGB
# lab = color.rgb2lab(img1).astype(np.float32)
lab = img1
print(lab)
# lab_t = transforms.ToTensor()(lab)
# print(lab)
# L = lab_t[[0], ...] / 50.0 - 1.0
# AB = lab_t[[1, 2], ...] / 110.0
L = lab[:, :, 0]
AB = lab[:, :, 1:3]
img_a = AB[:, :, 0]
img_b = AB[:, :, 1]
a = img_a.flatten()
b = img_b.flatten()
L = L.flatten()
# a = img_a.flatten()
# b = img_b.flatten()
# print(a, b)
za = stats.mode(a)[0][0]
zb = stats.mode(b)[0][0]
zl = stats.mode(L)[0][0]
print(zl, za, zb)
# LAB = torch.cat((L, AB), dim=0)
# print(L, LAB)
# 按R、G、B三个通道分别计算颜色直方图
# 显示3个通道的颜色直方图
# plt.plot(b_hist, label='B', color='blue')
# plt.plot(g_hist, label='G', color='green')
# plt.plot(r_hist, label='R', color='red')
# plt.legend(loc='best')
# plt.xlim([0, 256])
# plt.show()
#
hist_a = cv2.calcHist([a], [0], None, [256], [0, 255])
hist_b = cv2.calcHist([b], [0], None, [256], [0, 255])
hist_L = cv2.calcHist([L], [0], None, [100], [0, 99])
plt.plot(hist_a)
plt.plot(hist_b)
plt.plot(hist_L)
plt.xlim([0, 255])
plt.show()