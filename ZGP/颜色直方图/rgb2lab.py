import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


import cv2
import numpy as np

# 读入图像
img_file = r"H:\Datasets\VOC2012\JPEGImages\2007_000042.jpg"#读取图片
img = cv2.imread(img_file, 1)
# 转换为Lab色彩空间
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 分离出a通道
L, a, b = cv2.split(lab)

# 显示a通道图像
cv2.imshow("a_channel", a)
cv2.imshow("b_channel", b)
cv2.imshow("L_channel", L)
# 等待用户按下任意按键退出程序
cv2.waitKey(0)
cv2.destroyAllWindows()
