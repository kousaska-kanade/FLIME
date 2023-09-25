import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCH = 100
N_CLASSES = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# #trainData = dsets.ImageFolder('/root/weapons/train', transform)
# #testData = dsets.ImageFolder('/root/weapons/val', transform)
#
# trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  #num_wockers之前是2
# testLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  #num_wockers之前是2


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.contiguous().view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


# model = VGG16(n_classes=N_CLASSES)
# if torch.cuda.device_count() >= 1:
#     print("Use", torch.cuda.device_count(), 'gpus')
#     model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
#     model.to(device)
#
# # Loss, Optimizer & Scheduler
# cost = tnn.CrossEntropyLoss().cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#
# # # Train the model
# # for epoch in range(EPOCH):
# #     train_loss_epoch = 0
# #     val_loss_epoch = 0
# #     train_corrects = 0
# #     val_corrects = 0
# #     # 对训练数据的迭代器进行迭代计算
# #     model.train()
# #     for step, (images, labels) in enumerate(trainLoader):
# #         images = images.cuda()
# #         labels = labels.cuda()
# #         output = Myvggc(b_x)  # CNN在训练batch上的输出
# #         loss = loss_func(output, b_y)  # 交叉熵损失函数
# #         pre_lab = torch.argmax(output, 1)
# #         optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
# #         loss.backward()  # 损失的后向传播，计算梯度
# #         optimizer.step()  # 使用梯度进行优化
# #         train_loss_epoch += loss.item() * b_x.size(0)
# #         train_corrects += torch.sum(pre_lab == b_y.data)
# #
# #     # 计算一个epoch的损失和精度
# #     train_loss = train_loss_epoch / len(trainData.targets)
# #     train_acc = train_corrects.double() / len(trainData.targets)
#
#
# for epoch in range(EPOCH):
#     avg_loss = 0
#     cnt = 0
#     train_loss_epoch = 0
#     val_loss_epoch = 0
#     train_corrects = 0
#     val_corrects = 0
#     model.train()
#     for step, (images, labels) in enumerate(trainLoader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = cost(outputs, labels)
#         pre_lab = torch.argmax(outputs, 1)
#         avg_loss += loss.data
#         cnt += 1
#         if step%10==0:
#             print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
#         loss.backward()
#         optimizer.step()
#         train_loss_epoch += loss.item() * images.size(0)
#         train_corrects += torch.sum(pre_lab == labels.data)
#         #print(train_corrects)
#
#         # 计算一个epoch的损失和精度
#     train_loss = train_loss_epoch / len(trainData.targets)
#     train_acc = train_corrects.double() / len(trainData.targets)
#     print("[E: %d] loss: %f, acc: %f" % (epoch, train_loss, train_acc))
#     scheduler.step(avg_loss)
#
#     # 计算在验证集上的表现
#     model.eval()
#     for step, (val_x, val_y) in enumerate(testLoader):
#         val_x, val_y = val_x.to(device), val_y.to(device)
#         output =model(val_x)
#         loss = cost(output, val_y)
#         pre_lab = torch.argmax(output, 1)
#         val_loss_epoch += loss.item() * val_x.size(0)
#         val_corrects += torch.sum(pre_lab == val_y.data)
#     # 计算一个epoch的损失和精度
#     val_loss = val_loss_epoch / len(testData.targets)
#     val_acc = val_corrects.double() / len(testData.targets)
#     print("[Epoch: %d] val_loss: %f, val_acc: %f" % (epoch, val_loss,val_acc))
#
# # Test the model
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in testLoader:
#          images = images.cuda()
#          outputs = model(images)
#          _, predicted = torch.max(outputs.data, 1)
#          total += labels.size(0)
#          correct += (predicted.cpu() == labels).sum()
#          print(predicted, labels, correct, total)
#          print("avg acc: %f" % (100 * correct / total))
#
# # Save the Trained Model
# torch.save(model.state_dict(), '/data/vgg-weapons2.pth')
