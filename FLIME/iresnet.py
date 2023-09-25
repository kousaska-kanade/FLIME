import torch
from torch import nn
from utils import norm_crop
from scrfd import SCRFD
import os.path as osp
import cv2
import argparse
import os
import os.path as osp
import numpy as np
import datetime
import random
from torchsummary import summary
import torch

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #print(self.features)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        # print(11111111111111111111111111111)
        # print(block.expansion)
        # print(11111111111111111111111111111)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        #print(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)
# = iresnet50()
#summary(nett,(1,28,28),batch_size=1,device="cpu")
# class LL:
#     def __init__(self, N=10):
#         #随机化等设置参数
#         os.environ['PYTHONHASHSEED'] = str(1)
#         torch.manual_seed(1)
#         np.random.seed(1)
#         random.seed(1)
#
#     def generatee(self, im_v):
#         net1 = iresnet50()
#         weight = 'assets/w600k_r50.pth'
#         net1.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
#         net1.eval().cuda()
#
#         # model-2
#         net2 = iresnet100()
#         weight = 'assets/glint360k_r100.pth'
#         net2.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
#         net2.eval().cuda()
#
#         assert len(im_v.shape) == 3
#
#         # print(kpss)
#
#         bboxes, kpss = self.detector.detect(im_v, max_num=1)
#
#         vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)  # 被攻击图片中心对齐
#
#         #att_img = att_img[:, :,::-1]  ##image[:,:,::-1]作用：对颜色通道做变换，将图片从RGB图片转成BGR图片，image[:, ::-1, :]作用：将图像进行左右翻转，image[::-1, :, :]作用：将图像上下颠倒
#         # 还有对数组和列表的切片操作是怎么进行的，例如[:,:,m:n]，[:,:,1]
#         # 与(cv2.COLOR_RGB2BGR)有什么区别？
#         vic_img = vic_img[:, :, ::-1]
#
#         # get victim feature
#         vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # 变换成张量，和上面是不是有异曲同工之处
#         vic_img.div_(255).sub_(0.5).div_(0.5)  # 像素值归一到[-1,1]，对应的激活函数不同
#         vic_feats = self.model.forward(vic_img)
#         vic_output = self.model(vic_img)
#         # vic_probs = F.softmax(vic_output).detach().numpy()[0]
#         # vic_labels = np.argmax(vic_output)
#         #print(vic_output.argmax(1))
#         # process input
#         #att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
#         #att_img.div_(255).sub_(0.5).div_(0.5)
#         #att_output = self.model(att_img)
#         # # att_probs = F.softmax(att_output).detach().numpy()[0]
#         # # att_labels = np.argmax(att_output)
#         # print('_________________________________________')
#         # print(att_output.argmax(1).item())
#         # print('_________________________________________')
#         # att_img_ = att_img.clone()  # 拷贝攻击图片？
#         # att_img.requires_grad = True  # 可以计算梯度
#         # # file = open("test.txt","w")
#         # # file.write(str(sum)+":")
#         # # file.write("\n")
#         # # file.write(str(vic_output.argmax(1).item()))
#         # # file.write("\n")
#         # # file.write(str(att_output.argmax(1).item()))
#         # # file.close
#         # for i in tqdm(range(self.num_iter)):  # 训练100轮？
#         #     # if :
#         #     #     break
#         #     self.model.zero_grad()  # 模型中参数梯度设置为0
#         #     adv_images = att_img.clone()
#         #
#         #     # get adv feature
#         #     adv_feats = self.model.forward(adv_images)  # 得到特征
#         #
#         #     # caculate loss and backward
#         #     # loss = torch.mean(torch.square(adv_feats - vic_feats))#输出特征均值差
#         #     loss = -(cos_simi(net1(att_img), net1(vic_img)) * 8 + cos_simi(net2(att_img), net2(vic_img)) * 2)
#         #     # print("loss.type")
#         #     # print(loss.type)
#         #     # print("loss")
#         #     # print(loss)
#         #     loss.backward(retain_graph=True)  # 反向传播，计算当前梯度
#         #
#         #     grad = att_img.grad.data.clone()
#         #     # print(grad.abs().mean(dim=[1,2,3],keepdim=True))
#         #     # print(grad.abs().mean(dim=[1, 2, 3], keepdim=True).shape)
#         #     # print(grad)
#         #     # print(grad.shape)
#         #     grad = grad / grad.abs().mean(dim=[1, 2, 3],
#         #                                   keepdim=True)  # 取绝对值后按照通道方向、样本的高、宽取绝对值？并保持维度不变？[1,3,112,112]都除以[[[[0.0024]]]]梯度都是除以一个相同的数，有什么意义？
#         #     # print(grad)
#         #     # print(grad.shape)
#         #     sum_grad = grad
#         #     att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (
#         #                 1 - self.mask)  # 变化扰动，可以改变self.alpha等试试有没有效果
#         #     att_img.data = torch.clamp(att_img.data, -1.0, 1.0)  # 将每个张量都限制在-1~1之间，限制扰动大小？
#         #     att_img = att_img.data.requires_grad_(True)
#         #     # get diff and adv img
#         #
#         # diff = att_img - att_img_  # 扰动差异大小
#         # diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2,
#         #                                                        0) * 127.5  # 首先detach()确保梯度无法改变，然后通过还原成numpy数组还原图片，127.5应该是之前逆推得到的，但不懂怎么算的
#         # diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP,
#         #                       borderValue=0.0)  # 扰动也通过同一放射矩阵进行同样的对齐算法
#         # diff_bgr = diff[:, :, ::-1]  # 统一扰动格式与原图格式
#         # adv_img = im_a + diff_bgr  # 添加扰动
#         # adv_output = self.model(adv_img)
#         # att_probs = F.softmax(att_output).detach().numpy()[0]
#         # att_labels = np.argmax(att_output)
#         # print(adv_output.argmax(1))
#         # file = open("test.txt","w")
#         # file.write("\n")
#         # file.write(str(adv_output.argmax(1).item()))
#         # file.close
#         return vic_feats
#     def set_cuda(self):
#         #是否使用gpu
#         self.is_cuda = True
#         self.device = torch.device('cuda')
#         torch.cuda.manual_seed_all(1)
#         torch.cuda.manual_seed(1)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#
#     def load(self, assets_path):  # 生成掩膜
#         detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
#         # print(assets_path)
#         ctx_id = -1 if not self.is_cuda else 0
#         detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
#         img_shape = (112, 112)
#         # 根据预训练权重训练模型
#
#         # model.
#         # load face mask
#         self.detector = detector
#
#
# # att = osp.join(iddir, '0.png')
# def main(args):
#     att = 'E:/CZY/baseline/assets/0.png'
#     tool = LL()
#     if args.device == 'cuda':
#         tool.set_cuda()
#     tool.load('assets')
#     origin_att_img = cv2.imread(att)
#     vic_feat2 = tool.generatee(origin_att_img)
#     #nett.forward(att)
#     print(vic_feat2)
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output', help='output directory', type=str, default='output/')
#     parser.add_argument('--device', help='device to use', type=str, default='cuda')
#     args = parser.parse_args()
#     main(args)