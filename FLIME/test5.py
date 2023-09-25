from PIL import Image
import numpy as np

image = Image.open('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/00002465.jpg')
image_arr = np.array(image)

a = [1,2,3]
b = [4,5,6]
c = [7,8,9]
d = [10,11,12]
e = [13,14,15]
f = [16,17,18]

w = np.array([[a,b,c],[d,e,f]])
print(w)

# print(image_arr.shape)
# print(image_arr)
print(w[0])
print(w[0][:, 0])