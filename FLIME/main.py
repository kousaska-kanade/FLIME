import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os
import json
from VGG16 import *

from skimage.segmentation import mark_boundaries

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from lime import lime_image

# def read_directory(directory_name):
#     for filename in os.listdir(directory_name):
#         print(directory_name+'/'+filename)
#         img = cv2.imread(directory_name+'/'+filename)
#         cv2.imshow('image',img)
#         cv2.waitKey(0)

#read_directory('D:/weapons/train/000')

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


#img = get_image('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/test2.jpg')
img = get_image(r'F:\Lime-pytorch\100\heli\298-4.jpg')

#img = get_image('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/data/test2.jpg')
shape = img.size
rows = shape[0]
cols = shape[1]
# print(rows)
# print(cols)
#img = img.resize((224,224))
plt.imshow(img)
plt.show()
#img = img.crop((75,100,800,250))
#print(type(img))

#img = img.resize((224,224), Image.ANTIALIAS)

#img = cv2.resize(224,224)
# plt.imshow(img)
# plt.show()

# resize and take the center part of image to what our model expects
#We need to convert this image to Pytorch tensor and also apply whitening as used by our pretrained model.
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        transforms.ToTensor(),
        normalize
    ])

    return transf

# def get_input_transform():
#     #normalize = transforms.Normalize(mean=[0])
#     transf = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ])

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

#model = models.inception_v3(pretrained=True)
#model = torch.load('')
model = models.resnet152(pretrained = False)
#model.load('')

#model = VGG16(n_classes=20)
model.load_state_dict(torch.load(r'F:\Lime-pytorch\data_model_299.pth'),False)
#model = torch.load(r'F:\Lime-pytorch\vgg-weapons1.pth')
# idx2label, cls2label, cls2idx = [], {}, {}
# with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
#     class_idx = json.load(read_file)
#     idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
#     cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
#     cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
# model = iresnet.iresnet50()
# weight = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/w600k_r50.pth'
# model.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
#model.eval()
# model = facenet(backbone='mobilenet', mode="predict")
# model.load_state_dict(torch.load(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\vgg-weapons1.pth', map_location=device), strict=False)
model.eval()
img_t = get_input_tensors(img)
#model.eval()
#x = model.predict(img_t)
#print(x)
logits = model(img_t)

probs = F.softmax(logits, dim=1)
probs5 = probs.topk(5)
# print(probs5[1][0].detach().numpy())
# print(probs5[0][0].detach().numpy())
#tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))

#print(1)
# We are getting ready to use Lime. Lime produces the array of images from original input image by pertubation algorithm.
# So we need to provide two things: (1) original image as numpy array (2) classification function that would take array of purturbed images as input and produce the probabilities for each class for each image as output.
#
# For Pytorch, first we need to define two separate transforms: (1) to take PIL image, resize and crop it (2) take resized, cropped image and apply whitening.
# def get_pil_transform():
#     trans = transforms.Compose([
#
#     ])
def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict, # classification function
                                         top_labels=3,
                                         hide_color=0,
                                         num_samples=1000) # number of images that will be sent to classification function
                                                            #produce database
print(explanation.top_labels[0])
print(explanation.top_labels[1])
print(explanation.top_labels)
plt.axis('off') # 去坐标轴
plt.xticks([]) # 去刻度
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
#plt.savefig('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/0.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()
cv2.waitKey(0)
#print(1)
#pylab.show()
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
#print(2)
plt.axis('off') # 去坐标轴
plt.xticks([]) # 去刻度
plt.imshow(img_boundry2)
#plt.savefig('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/1.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
#img_boundry2.show()
#print(3)
plt.show()
cv2.waitKey(0)

ind = explanation.top_labels[0]
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()


#print(4)