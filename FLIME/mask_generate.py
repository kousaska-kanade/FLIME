#生成
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img = np.zeros([160, 160, 3], np.uint8)
img.fill(255)
file_origin_path = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/0.png'
file_new_path = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/1.png'
file_masko_path = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/masko.png'
file_maskn_path = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/maskn.png'
cv2.imwrite(file_masko_path,img)
img_origin = Image.open(file_origin_path)
img_new = Image.open(file_new_path)
img_origin.convert('RGB')
img_new.convert('RGB')
img_origin = img_origin.resize((160,160))
plt.imshow(img_origin)
plt.show()
cv2.waitKey(0)
img_new = img_new.resize((160,160))
plt.imshow(img_new)
plt.show()
cv2.waitKey(0)
mask = Image.open(file_masko_path)
np_origin = np.array(img_origin)
np_new = np.array(img_new)
np_mask = np.array(mask)
# img_origin = cv2.imread(file_origin_path)
# img_new = cv2.imread(file_new_path)
#shape = img_origin.shape
# for i in range(0,shape[0]):
#     for j in range(0,shape[1]):
#         if np_origin[i,j] != np_new[i,j]:
#             #np_origin = np.array(file_origin_path)
#             #np_new = np.array(file_new_path)
#             np_mask[i,j] = 0
for h in range(img_origin.height):
    for w in range(img_origin.width):
        if(img_origin.getpixel((w, h)) != img_new.getpixel((w, h))):

            mask.putpixel((w, h), (0,0,0))
mask = mask.convert("RGB")
mask.save(file_maskn_path)