#生成
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
file_origin_path = r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\20.JPG'
file_high_path = r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\high_test.jpg'
# img_origin = cv2.write(file_origin_path)
# img_array = np.array(img_origin)
img_origin = Image.open(file_origin_path)
img_origin = img_origin.resize((250,250))
img_origin2 = img_origin.crop((50, 40, 200, 190))#左、上、右、下
plt.imshow(img_origin)
plt.show()
cv2.waitKey(0)
img_high = Image.open(file_high_path)
img_high = img_high.resize((250,250))
img_high = img_high.crop((50, 40, 200, 190))
plt.imshow(img_origin2)
plt.show()
cv2.waitKey(0)
#np_origin = np.array(img_origin)
#ps = img_origin.shape
count = 0
file_path1 = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/crop1/'
file_path2 = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/crop2/'
#img = cv2.imread(file_path + str(i) + '.jpg')
for i in range(20):
    for j in range(20):
        # print(img)
        img_temp = img_origin2.crop((i, j, i + 130, j + 130))
        img_temp2 = img_high.crop((i, j ,i + 130, j + 130))
        count = count + 1
        img_temp = img_temp.convert("RGB")
        img_temp.save(file_path1 + str(count) + '.jpg')
        img_temp2.save(file_path2 + str(count) + '.jpg')

directory_name = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/crop2'
count2 = 0
count3 = 0
maxi = 0
flag = 0
print("1")
for filename in os.listdir(directory_name):
        #print(filename)  # 仅仅是为了测试
        count2 = count2 + 1
        img = cv2.imread(directory_name + "/" + str(count2) + '.jpg')
        img3 = Image.open(directory_name + "/" +str(count2) + '.jpg')
        #####显示图片#######
        #cv2.imshow(filename, img)
        #cv2.waitKey(0)
        ps = img.shape
        img_array = np.array(img3)
        for i in range(ps[1]):
            for j in range(ps[0]):
                # print(img)
                if (img_array[i, j] > 10):
                    # 缺一个元祖存图片id
                    count3 = count3 + 1
        if count3 > maxi:
            maxi = count3
            flag = count2
print(flag)

