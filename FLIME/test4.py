import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
from PIL import Image
from VGG16 import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 对比展现原始图片和对抗样本图片
def show_images_diff(original_img, original_label, adversarial_img, adversarial_label):
    import matplotlib.pyplot as plt
    plt.figure()

    # 归一化
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255.0
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(131)
    plt.title('原始：主战坦克')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('0.07的扰动')
    difference = adversarial_img - original_img
    # (-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')


    plt.subplot(133)
    plt.title('攻击后：榴弹炮')
    plt.imshow(adversarial_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像加载以及预处理
image_path = "D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/00004383.jpg"
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (224, 224))
img = orig.copy().astype(np.float32)
mask_path = "D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/mask.png"
mask = cv2.imread(mask_path)[..., ::-1]
mask = cv2.resize(mask, (224, 224))
mask2 = mask.copy().astype(np.float32)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mask2 /=255
mask2 = mask2.transpose(2, 0, 1)
mask2 = np.expand_dims(mask2, axis=0)
mask2 = Variable(torch.from_numpy(mask2).to(device).int())
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)

img = np.expand_dims(img, axis=0)

img = Variable(torch.from_numpy(img).to(device).float())
print(img.shape)

# 使用预测模式 主要影响droupout和BN层的行为
#model = models.alexnet(pretrained=True).to(device).eval()
N_CLASSES = 20
model = torch.nn.DataParallel(VGG16(n_classes=N_CLASSES), device_ids=[0])
model.load_state_dict(torch.load('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/vgg-weapons1.pth'))
model.eval()

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

epochs = 10  # 训练轮次，只需一次，多了就变成I-FGSM了
e = 0.001  # 扰动值

target = 5  # 此处是一个定向攻击
target = Variable(torch.Tensor([float(target)]).to(device).long())  # 转换数据类型
torch.set_printoptions(profile="full")
#print(1-mask2)
for epoch in range(epochs):

    # forward + backward
    output = model(img)

    loss = loss_func(output, target)
    label = np.argmax(output.data.cpu().numpy())

    print("epoch={} loss={} label={}".format(epoch, loss, label))

    # 如果定向攻击成功
    if label == target:
        print("成功")
        break

    # 梯度清零
    optimizer.zero_grad()
    # 反向传递 计算梯度
    loss.backward()
    #print(e * torch.sign(img.grad.data) * (1 - mask2.data))
    img.data = img.data - e * torch.sign(img.grad.data)*(1-mask2.data) # FGSM最重要的公式
print(model(img).argmax())
adv = img.data.cpu().numpy()[0]
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
adv = np.clip(adv, 0, 255).astype(np.uint8)  # np数据类型是0-255，转PIL之后进行存储
# 对抗样本的保存
# print(adv.shape)
# img = Image.fromarray(adv)
# img.show()
# plt.imshow(img)
# plt.show()
# img.save('one.jpg')#图片存储为什么不用指定路径，存储结果在远程服务器
show_images_diff(orig, 13, adv, target.data.cpu().numpy()[0])