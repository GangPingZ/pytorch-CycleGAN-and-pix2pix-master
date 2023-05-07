import cv2
import numpy as np


# RGB分别计算SSIM再取平均值
def ssim(img1, img2):
    # 分离颜色通道
    img1_channels = cv2.split(img1)
    img2_channels = cv2.split(img2)

    # 初始化SSIM指标为0
    ssim_index = 0

    # 计算每个颜色通道的SSIM指标
    for i, channel in enumerate(img1_channels):
        img1_mean, img1_std = cv2.meanStdDev(channel)
        img2_mean, img2_std = cv2.meanStdDev(img2_channels[i])
        covar = np.cov(channel.ravel(), img2_channels[i].ravel())[0][1]
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        numerator = (2 * img1_mean * img2_mean + c1) * (2 * covar + c2)
        denominator = (img1_mean ** 2 + img2_mean ** 2 + c1) * (img1_std ** 2 + img2_std ** 2 + c2)
        ssim_index += numerator / denominator

    # 取平均值
    ssim_index /= 3

    return ssim_index

# 读取两个图像
img1 = cv2.imread(r"H:\SMU\Wang\1080ti\EUVP_LAB\test_latest\images\test_p0__real_A.png")
img2 = cv2.imread(r"H:\SMU\Wang\1080ti\EUVP_LAB\test_latest\images\test_p0__fake_B.png")

# 计算SSIM指标
ssim_index = ssim(img1, img2)

print("SSIM指标值为:", ssim_index)

# 计算灰度图像的SSIM，指标会更高。
# def ssim(img1, img2):
#     # 计算均值和方差
#     img1_mean, img1_std = cv2.meanStdDev(img1)
#     img2_mean, img2_std = cv2.meanStdDev(img2)
#
#     # 计算协方差
#     covar = np.cov(img1.ravel(), img2.ravel())[0][1]
#
#     # 计算常数
#     c1 = (0.01 * 255) ** 2
#     c2 = (0.03 * 255) ** 2
#
#     # 计算SSIM指标
#     numerator = (2 * img1_mean * img2_mean + c1) * (2 * covar + c2)
#     denominator = (img1_mean ** 2 + img2_mean ** 2 + c1) * (img1_std ** 2 + img2_std ** 2 + c2)
#     return numerator / denominator
#
# # 读取两个图像
# img1 = cv2.imread(r"H:\SMU\dataset\EUVP\test_samples-20230425T075315Z-001\test_samples\trainA\test_p5_.jpg")
# img2 = cv2.imread(r"H:\SMU\dataset\EUVP\test_samples-20230425T075315Z-001\test_samples\trainB\test_p5_.jpg")
#
# # 转换为灰度图像
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# # 计算SSIM指标
# ssim_index = ssim(img1_gray, img2_gray)
#
# print("SSIM指标值为:", ssim_index)
