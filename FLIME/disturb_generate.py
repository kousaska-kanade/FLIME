import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import iresnet
import torch.nn.functional as F
epsilons = [0, .05, .1, .15, .2, .25]
pretrained_model = ''
use_cuda = True
file_maskn_path = r'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/result/maskn.png'
#数据集加载
device = 'cuda'
#data_loader = torch.utils.data.DataLoader()

mask = Image.open(file_maskn_path)
np_mask = np.array(mask)
def especial_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad * (1 - np_mask)#要改成根据颜色分配不同扰动大小，要不要设计一个loss？
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test( model, device, test_loader, epsilon ):

    # 精度计数器
    correct = 0
    adv_examples = []

    # 循环遍历测试集中的所有示例
    for data, target in test_loader:

        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)

        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True

        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            continue

        # 计算损失
        loss = F.nll_loss(output, target)

        # 将所有现有的渐变归零
        model.zero_grad()

        # 计算后向传递模型的梯度
        loss.backward()

        # 收集datagrad
        data_grad = data.grad.data

        # 唤醒FGSM进行攻击
        perturbed_data = especial_attack(data, epsilon, data_grad)

        # 重新分类受扰乱的图像
        output = model(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # 保存0 epsilon示例的特例
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # 稍后保存一些用于可视化的示例
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # 计算这个epsilon的最终准确度
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确性和对抗性示例
    return final_acc, adv_examples


if __name__ == '__main__':
    # model = iresnet.iresnet50()
    # weight = 'D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/w600k_r50.pth'
    # model.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    # test(model, epsilons, )
    especial_attack(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\10-14.jpg', 0.03, -1)