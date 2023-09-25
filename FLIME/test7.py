# import cv2
# import math
# file_path = r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\00002465.jpg'
# img = cv2.imread(file_path)
#
# sp = img.shape
# print(sp)
# #print(sp[0])
# #划分大小
# height = 10
# width = 10
#
# #划分成的块数
# h_num = math.ceil(sp[0] / height)
# w_num = math.ceil(sp[1] / width)
#
# print(h_num)
# print(w_num)
#
# #对块编号
# h = 1
# w = 1

# for i in range(h_num):
#         for j in range(w_num):
#             # print(i,j)
#             x = int(i * crop_h)  #高度上裁剪图像个数
#             y = int(j * crop_w)
#             print(x,y)
#             img_crop = img_new[x : x + crop_h,y : y + crop_w]
#             # print(z)
#             saveName= name.split('.')[0] + '-' + str(i) +'-'+ str(j) +".png"  #小图像名称，内含小图像的顺序
#             cv2.imwrite(saveDir+saveName,img_crop)


# for i in range((w - 1) * width,(w) *width):
#     for j in range((h - 1) * height,(h) * height):
# for i in range(h_num):
#     for j in range(w_num):
#         x = int(i * h)
#         y = int(j *w)
#         img_crop = img_new
import math

from PIL import Image
import numpy as np
import cv2

file_path = r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\high_test.jpg'
file_path2 = r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\Adolfo_Rodriguez_Saa_0002.jpg'
img = cv2.imread(file_path)
img2 = cv2.imread(file_path2)
sp = img.shape
print(sp)
#print(sp[0])
#划分大小
height = 10
width = 10

#划分成的块数
h_num = math.ceil(sp[0] / height)
w_num = math.ceil(sp[1] / width)

constant = cv2.copyMakeBorder(img, 0, (h_num * height) - sp[0], 0, (w_num * width) - sp[1], cv2.BORDER_CONSTANT, value=0)
constant2 = cv2.copyMakeBorder(img2, 0, (h_num * height) - sp[0], 0, (w_num * width) - sp[1], cv2.BORDER_CONSTANT, value=0)
cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/constant.jpg',constant)
cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/constant2.jpg',constant2)
#图片填充
file_path = r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\high_test.jpg'
file_path2 = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/constant2.jpg'
img = Image.open(file_path)
img2 = Image.open(file_path2)
imgSize = img.size  # 大小/尺寸
w = img.width  # 图片的宽
h = img.height  # 图片的高
f = img.format  # 图像格式


#算数量
w_num = math.ceil(w/width)
h_num = math.ceil(h/height)

count = 0
#还少一个填充吧，不知道填充不填充影响结果吗？
#图片划分
for i in range(h_num):
    for j in range(w_num):
        img_new = img.crop((j * width, i * height, ((j + 1) * width), ((i + 1) * height)))
        img_new.save('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/crop/'+str(count)+'.jpg')
        count = count + 1

count3 = 0
pic_list = []
#把高频灰度图传过来·
for i in range(count):
    file_path = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/crop/'
    img = cv2.imread(file_path + str(i) + '.jpg')
    img3 = Image.open(file_path + str(i) + '.jpg')
    img3.convert('L')
    img_array = np.array(img3)
    ps = img.shape
    #print(ps)
    count2 = 0
    #print(type(img))
    #print(img.shape)
    for j in range(ps[1]):
        for k in range(ps[0]):
            #print(img)
            if(img_array[j, k] > 10):
                #缺一个元祖存图片id
                count2 = count2 + 1
    if(count2 > 20):
        pic_list.append(i)
        count3 = count3 + 1
print(count3)
print(pic_list)

for i in pic_list:
    print(i)
    for j in range((((i) // 16)) * 10, (((i) // 16) + 1) * 10):
        for k in range(((i) % 16) * 10, (((i) % 16) + 1) * 10):
            #print(j,k)
            img2.putpixel((k, j),(0, 0, 0))
img3 = img2.convert("RGB")
img3.save('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/test.jpg')

file_path = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/0.png'
img4 = Image.open(file_path)

for i in range(img4.width):
    for j in range(img4.height):
        if(img3.getpixel((i, j)) != (0, 0, 0)):
            img4.putpixel((i, j),(0, 0, 0))
img4 = img4.convert("RGB")
img4.save('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/test2.jpg')
# file_path = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/test.jpg'
# file_path2 = ''
# img = Image.open(file_path)
# img2 = Image.open(file_path2)
# img = img.load()
# for i in range():
#     for j in range():
#         if(img[j,i] == (0,0,0)):
#             img2[j,i] = (255,255,255)

#怎么让像素值乘一个固定值？
#这里可以用for循环，像素值如果为（0,0,0）就不变，不然就置（0,0,0）
# for i in range(h):
#     for j in range(w):
        #for k in pic_list:
        #if((i // 20) + (j // 20) == k):
            #print(i)
                #print(k)
        #j = j +20
    #i = i + 20
#
# print(imgSize)
# print(w, h, f)