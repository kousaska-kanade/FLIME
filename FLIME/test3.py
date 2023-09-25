from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import cv2
from torch.autograd import Variable
e=0.5#扰动值
#获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "data/goldfish.jpg"
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)
# 使用Imagenet的均值和标准差是一种常见的做法。它们是根据数百万张图像计算得出的。如果要在自己的数据集上从头开始训练，则可以计算新的均值和标准差，这是一种经验值
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 归一化，为什么进行归一化，因为训练效果好
img /= 255.0
img = (img - mean) / std
# 把HWC的图片转为CHW的图片
img = img.transpose(2, 0, 1)

img = np.expand_dims(img, axis=0)

img = Variable(torch.from_numpy(img).to(device).float())
print(img.shape)

# 使用预测模式 主要影响droupout和BN层的行为
model = models.alexnet(pretrained=True).to(device).eval()
# 取真实标签
label = np.argmax(model(img).data.cpu().numpy())  # 这里为什么要加cup（）？因为np无法直接转为cuda使用，要先转cpu
print("label={}".format(label))
# 图像数据梯度可以获取
img.requires_grad = True

# 设置为不保存梯度值 自然也无法修改
for param in model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam([img])  # 优化器
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵计算损失

epochs = 10  # 训练轮次
target = 31  # 原始图片的标签
target = Variable(torch.Tensor([float(target)]).to(device).long())  # 转换数据类型
print(target)

def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon*sign_data_grad
    #噪声越来越大，机器越来越难以识别，但人眼可以看出差别
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image