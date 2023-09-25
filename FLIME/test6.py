# -*- coding: utf-8 -*-
import cv2
import cv2 as cv
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt


def filter_high_f(fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift


def filter_low_f(fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg


def get_low_high_f(img, radius_ratio):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    hight_parts_fshift = filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = filter_high_f(fshift.copy(), radius_ratio=radius_ratio)

    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


if __name__ == '__main__':
    radius_ratio = 0.5  # 圆形过滤器的半径：ratio * w/2
    img = cv.imread(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\20-4.JPG', cv2.IMREAD_GRAYSCALE)
    low_freq_part_img, high_freq_part_img = get_low_high_f(img, radius_ratio=radius_ratio)  # multi channel or single
    cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/high_test.jpg', high_freq_part_img)
    print(type(high_freq_part_img))
    print(high_freq_part_img.shape)
    print(high_freq_part_img)
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(low_freq_part_img, 'gray'), plt.title('low_freq_img')
    plt.axis('off')
    plt.subplot(133), plt.imshow(high_freq_part_img, 'gray'), plt.title('high_freq_img')
    #plt.savefig('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/high_test.jpg')
    plt.axis('off')
    plt.show()

#没超过一定像素值的全部变成黑色
#拿高频图片进行一步一步晒，小于阈值的话，原图变黑