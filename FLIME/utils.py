import os

import numpy as np
import cv2
from skimage import transform as trans
import torch
import torch.nn.functional as F
from scrfd import SCRFD
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    _src = float(image_size)/112 * arcface_src
    tform.estimate(lmk, _src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    #img 250 250 3
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped, M

def cos_simi(emb_before_pasted, emb_target_img):
    """
    :param emb_before_pasted: feature embedding for the generated adv-makeup face images
    :param emb_target_img: feature embedding for the victim target image
    :return: cosine similarity between two face embeddings
    """
    return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                      / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))


def cal_target_loss(self, before_pasted, target_img, model_name):
    """
    :param before_pasted: generated adv-makeup face images
    :param target_img: victim target image
    :param model_name: FR model for embedding calculation
    :return: cosine distance between two face images
    """

    # Obtain model input size
    input_size = self.models_info[model_name][0][0]
    # Obtain FR model
    fr_model = self.models_info[model_name][0][1]

    before_pasted_resize = F.interpolate(before_pasted, size=input_size, mode='bilinear')
    target_img_resize = F.interpolate(target_img, size=input_size, mode='bilinear')

    # Inference to get face embeddings
    emb_before_pasted = fr_model(before_pasted_resize)
    emb_target_img = fr_model(target_img_resize).detach()

    # Cosine loss computing
    cos_loss = 1 - self.cos_simi(emb_before_pasted, emb_target_img)

    return cos_loss

def cal_eva(ori,adv,oriBbox):
    # 8连通计算扰动距离
    # im1:(250,250,3)
    # im2:(250,250,3)
    # return:扰动距离

    xmin, ymin, xmax, ymax = round(oriBbox[0][0]), round(oriBbox[0][1]), round(oriBbox[0][2]), round(oriBbox[0][3])
    faceArea = (oriBbox[0][3]-oriBbox[0][1])*(oriBbox[0][2]-oriBbox[0][0])

    #获得修改域
    oriFace = ori[ymin:ymax, xmin:xmax, :]
    advFace = adv[ymin:ymax, xmin:xmax, :]
    diff = advFace - oriFace
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 膨胀操作
    bin_clo = cv2.dilate(gray, kernel2, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)#8连通

    #计算扰动区域面积
    advArea = 0
    for i in range(1,num_labels):
        advArea += stats[i][4]
    return advArea/faceArea*100.0
