import pickle
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2
from PIL import ImageFile
import matplotlib.pyplot as plt
from PIL import Image
from VGG16 import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
N_CLASSES = 20
model = torch.nn.DataParallel(VGG16(n_classes=N_CLASSES), device_ids=[0])
model.load_state_dict(torch.load('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/vgg-weapons1.pth'))
model.eval()

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)
#img = get_image('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/img.png')

img = get_image('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/00013737.jpg')
shape = img.size
rows = shape[0]
cols = shape[1]
# print(rows)
# print(cols)
img = img.crop((rows/2-350,cols/2-125,rows/2+350,cols/2+50))
#images = cv2.imread('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/00037826.jpg')
img_t = get_input_tensors(img)
outputs = model(img_t)
#_, predicted = torch.max(outputs.data, 1)
#total += labels.size(0)
#correct += (predicted.cpu() == labels).sum()
#print(predicted)

plt.imshow(img)
#plt.title(predicted, loc='center')
plt.show()