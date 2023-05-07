import argparse
import glob
import os
import cv2
import numpy as np
# from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


#z SSIM这里不需要区分real图片、fake图片；这里是多通道的SSIM代码
def compare_ssim(img1, img2):
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

    return float(ssim_index)

def calc_measures(hr_path, txt_path, image_format="png", calc_psnr=True, calc_ssim=True):
    # z 结果的txt文件保存路径
    txt_path = txt_path + "mean_ssim_psnr.txt"
    if not os.path.exists(txt_path):
        os.system(r"touch {}".format(txt_path))  # 调用系统命令行来创建文件
    #z 'w'对已存在的文件覆盖
    txt_file = open(txt_path, 'w')
    HR_files = [i for i in glob.glob(hr_path + '/*') if i.endswith("real." + image_format)]
    mean_psnr = 0
    mean_ssim = 0

    for file in HR_files:
        hr_img = cv2.imread(file)
        filename = file.rsplit('/', 1)[-1]
        path = os.path.join(args.inference_result, filename)
        path = path.replace("real.", "fake.")

        if not os.path.isfile(path):
            raise FileNotFoundError('')

        inf_img = cv2.imread(path)

        print('-' * 10)
        if calc_psnr:
            psnr = compare_psnr(hr_img, inf_img)
            # print('{0} : PSNR {1:.4f} dB'.format(filename, psnr))
            mean_psnr += psnr
        if calc_ssim:
            ssim = compare_ssim(hr_img, inf_img)  # 三通道SSIM比较值
            # print('{0} : SSIM {1:.4f}'.format(filename, ssim))
            mean_ssim += ssim
        print('{0} : PSNR {1:.4f} dB, SSIM {2:.4f}'.format(filename, psnr, ssim))
        txt_file.write('{0} : PSNR {1:.4f} dB, SSIM {2:.4f}\n'.format(filename, psnr, ssim))

    print('-' * 10)
    if calc_psnr:
        M_psnr = mean_psnr / len(HR_files)
    if calc_ssim:
        M_ssim = mean_ssim / len(HR_files)
    print('mean-PSNR: {0:.4f} dB, mean-SSIM: {1:.4f}'.format(M_psnr, M_ssim))
    txt_file.write('mean-PSNR: {0:.4f} dB, mean-SSIM: {1:.4f}'.format(M_psnr, M_ssim))
    txt_file.write('\n'*2)
    txt_file.close()


def calc_single_image_measures():
    hr_img_path = args.HR_data_dir
    inf_img_path = args.inference_result
    hr_img = cv2.imread(hr_img_path)
    inf_img = cv2.imread(inf_img_path)
    psnr = compare_psnr(hr_img, inf_img)
    ssim = compare_ssim(hr_img, inf_img)
    print('PSNR: {0:.4f} dB，SSIM: {1:.4f}'.format(psnr, ssim))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_data_dir', default=r"H:\SMU\Wang\RCA-CycleGAN\RCA-CycleGAN\results\RCA\test_latest\images", type=str)       #原始图像路径
    parser.add_argument('--inference_result', default=r"H:\SMU\Wang\RCA-CycleGAN\RCA-CycleGAN\results\RCA\test_latest\images", type=str)  #生成图像路径
    args = parser.parse_args()

    #z 计算多张图片的指标
    calc_measures(args.HR_data_dir, txt_path=r"H:\SMU\Wang\RCA-CycleGAN\RCA-CycleGAN\results\RCA\test_latest\images", calc_psnr=True, calc_ssim=True)

    #z 计算单张图片的指标；此时上面的路径就是图片的路径
    # calc_single_image_measures()


