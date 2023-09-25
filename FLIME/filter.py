import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(r"D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\00013737.jpg",0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(241), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(242), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
################################################################
###############            进行低通滤波             #############
################################################################
#设置低通滤波器
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2) #中心位置
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

#掩膜图像和频谱图像乘积
f = dft_shift * mask
print(f.shape, dft_shift.shape, mask.shape)

#傅里叶逆变换
ishift = np.fft.ifftshift(f)
iimg = cv2.idft(ishift)
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

#显示低通滤波处理图像
plt.subplot(243), plt.imshow(res, 'gray'), plt.title('High frequency graph')
plt.axis('off')
plt.savefig('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/high.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
###################################################################



#######################################################################
###################          进行高通滤波         ######################
#######################################################################
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

#设置高通滤波器
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

#傅里叶逆变换
ishift = np.fft.ifftshift(fshift)
himg = np.fft.ifft2(ishift)
himg = np.abs(himg)
# plt.subplot(244)
# plt.imshow(himg,'gray')
#显示原始图像和高通滤波处理图像
plt.subplot(244), plt.imshow(himg, 'gray'), plt.title('Low frequency graph')
plt.savefig('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/low.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.axis('off')
plt.show()
