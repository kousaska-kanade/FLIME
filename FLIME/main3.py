import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.applications import ResNet152 as resn
from keras.preprocessing import image
import tensorflow as tf
from skimage.io import imread
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt
from skimage.util import img_as_float
import os,sys
#%matplotlib inline
import numpy as np
import cv2
import lime
from lime import lime_image
#from lime import lime_base
from skimage.segmentation import mark_boundaries
from model_resnet152 import *
#physical_devices = tf.config.experimental.list_physical_devices('CPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
import ace_helpers
from gradcam import TARGET_SIZE
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#inet_model = tf.keras.applications.resnet.ResNet152(input_shape=(224,224,3), weights=r'F:\Lime-pytorch\resnet152_xin1.h5', classes=10)
#inet_model.load_weights(r"D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
inet_model = tf.keras.applications.resnet.ResNet152(include_top=True,input_shape=(224,224,3),classes=10,weights=None)
inet_model.load_weights(r'F:\Lime-pytorch\resnet152_xin1.h5')

mainFolder = r"F:\Lime-pytorch\100\imagenet"
myFolders = os.listdir(mainFolder)


count = 0

# def preprocess_image(image_file,inshape):
#     image = tf.io.read_file(image_file)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, inshape) # 将图片尺寸调整为模型期望的大小
#     # image = tf.cast(image, tf.float32)
#     # mean = [0.485, 0.456, 0.406]
#     # std = [0.229, 0.224, 0.225]
#     # image = tf.image.per_image_standardization(image - mean) / std
#     #image = tf.image.rgb_to_grayscale(image)
#     image = tf.expand_dims(image, axis=0) # 添加批次维度
#     image = tf.cast(image, tf.float32) / 255.0 # 归一化处理
#     #out.append()
#     return image.numpy()

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        #x = inc_net.preprocess_input(x)
        #x =tf.keras.applications.resnet.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

# #target_size = (model.input.shape[1], model.input.shape[2])
# def preprocess_image(image_file,inshape):
#     image = tf.io.read_file(image_file)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, inshape) # 将图片尺寸调整为模型期望的大小
#     # image = tf.cast(image, tf.float32)
#     # mean = [0.485, 0.456, 0.406]
#     # std = [0.229, 0.224, 0.225]
#     # image = tf.image.per_image_standardization(image - mean) / std
#     image = tf.expand_dims(image, axis=0) # 添加批次维度
#     image = tf.cast(image, tf.float32) / 255.0 # 归一化处理
#     return image

for folder in myFolders:
    path = mainFolder + '/' + folder
    count = count + 1
    print(count)
    #print(r'F:\Lime-pytorch\result-data2' + folder + "1")
    # img = cv2.imread(r'D:\PyCharm 2021.1.3\pycharmproject\tiny-imagenet-200\tiny-imagenet-200\val\images' + "/" +pic_name, cv2.IMREAD_UNCHANGED)
    #images = preprocess_image(r'F:\Lime-pytorch\100\heli\00006861.jpg',(224,224))
    #print(images.shape());
    images2 = transform_img_fn([os.path.join('data', path)])
    #print(images2[0].shape)
    # image3 = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\superpixel-result3\0.jpg')])
    # images = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img.png')])
    # image2 = cv2.imread(r'F:\Lime-pytorch\100\plane\220-2.jpg')
    # image2 = cv2.resize(image2,(224,224))
    # print(images.shape)
    # image3 = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.
    # 1.3\pycharmproject\Lime-pytorch\img3.png')])
    # I'm dividing by 2 and adding 0.5 because of how this Inception represents images

    #plt.imshow(images2[0] / 2 + 0.5)
    #inet_model.
    preds = inet_model.predict(images2)
    #print(preds[0].shape())
    #print(type(images))
    #print(images2[0].shape)
    #print(images2.shape)
    # plt.imshow(images2)
    # plt.show()
    #plt.imshow(images2[0])
    #plt.show()
    # preds2 = inet_model.predict(image3)
    #print(decode_predictions(preds)[0][0][1])
    # print(preds2[0,239])
    # 239
    # for x in decode_predictions(preds)[0]:
    #     print(x)
    # print(preds)
    # preds_classes = np.argmax(preds)
    # print(preds_classes)
    # 创建一个解释器
    explainer = lime_image.LimeImageExplainer()

    # 创建结构关系矩阵
    # n = 4
    # block = np.zeros(n)
    # relation = np.zeros((n, n))
    mean_value = np.mean(images2[0])
    # 设置屏蔽颜色
    explanation = explainer.explain_instance(images2[0].astype('double'), inet_model.predict, top_labels=2, hide_color=mean_value,
                                             num_samples=500)
    print(explanation.top_labels[0])
    print(explanation.top_labels)
    # used_features, neighborhood_data, easymodel = explainer.get_model(images[0].astype('double'), inet_model.predict, num_features=2, top_labels=5, hide_color=0, num_samples=100)
    # print(neighborhood_data[0, [1, 4, 6]])
    # print(used_features)
    # print(neighborhood_data[:, used_features])
    # print(neighborhood_data[1])
    # print("简易模型输出：")
    # preds2 = easymodel.predict(images.reshape(1, -1))
    # for x in decode_predictions(preds2)[0]:
    #     print(x)
    # images = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\superpixel-result\18.jpg')])
    # preds3 = easymodel.predict(images.reshape(1, -1))
    # for x in decode_predictions(preds3)[0]:
    #     print(x)
    # easymodel.predict()
    # print(explanation.top_labels)
    # print(explanation.local_exp)
    # print(explanation.segments.shape)
    # print(images.shape)

    # fig = plt.figure('Superpixels')
    # ax = fig.add_subplot(1, 1, 1)
    # #ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)), explanation.segments))
    # # plt.axis("off")
    # plt.show()
    # print("------------------------------------------------------------------------------")
    # print(explanation.predict_proba)
    # temp, mask = explanation.get_image_and_mask(285, positive_only=True, num_features=2, hide_rest=True)
    # plt.imshow(images,mask)
    # plt.show()
    # count = 0
    # print(enumerate(np.unique(explanation.segments)))
    #count = 0
    # for (i, segVal) in enumerate(np.unique(explanation.segments)):
    #     # print("###################################################################################################################")
    #     # print(explanation.segments)
    #     # print("###################################################################################################################")
    #     #print(segVal)
    #     mask = np.zeros(image2.shape[:2], dtype="uint8")
    #     #mask[:] = 255;
    #     mask[explanation.segments == segVal] = 255
    #     mask[explanation.segments == segVal + 23] = 255
    #     #mask[explanation.segments == segVal + 7] = 255
    #     #mask[explanation.segments == segVal + 9] = 255
    #     #mask[explanation.segments == segVal + 11] = 255
    #     #mask[explanation.segments == segVal + 35] = 255
    #     # mask[segments == segVal + 3] = 255
    #     # mask[segments == segVal + 4] = 255
    #     cv2.imshow("Mask", mask)
    #     #cv2.waitKey(0)
    #     #cv2.imshow("Applied", np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
    #     #print(mask.shape)
    #     #cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result2/' + str(count) + '.jpg', np.multiply(image2, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) == 0))
    #     cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result5/' + str(count) + '.jpg', np.multiply(image2, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) != 0))
    #     count = count + 1

    # #获取贡献值
    # for i in range(n):
    #     mask = np.zeros(image2.shape[:2], dtype="uint8")
    #     mask[explanation.segments == block[i]] = 255
    #     cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result/' + str() + '.jpg', np.multiply(image2, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
    #
    # for i in range(n):
    #     for j in range(n):
    #         if i != j:
    #             mask = np.zeros(image2.shape[:2], dtype="uint8")
    #             mask[explanation.segments == block[i]] = 255
    #             mask[explanation.segments == block[j]] = 255
    #             cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result2/' + str() + '.jpg', np.multiply(image2, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
    #
    # #矩阵保存关联值
    # for i in range(n):
    #     for j in range(n):
    #         if i == j:
    #             relation[i][j] = 1
    #         else:
    #             images1 = cv2.imread(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img.png')
    #             images2 = cv2.imread(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img.png')
    #             imagesall = cv2.imread('')
    #             preds1 = inet_model.predict(images1)
    #             preds2 = inet_model.predict(images2)
    #             preds3 = inet_model.predict(imagesall)
    #             rela = preds3 - preds2 - preds1
    #             relation[i][j] = rela

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=True)
    # #print("111111111111111111111111")
    # #plt.imshow(temp / 2 + 0.5)
    # # print(temp.shape)
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.show()

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2,
                                                hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask, mode='outer'))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'F:\Lime-pytorch\019result/' + str(count) + "-1", bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.show()
    # cv2.waitKey(0)

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=3, hide_rest=False)
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.show()

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=3, hide_rest=False, min_weight=0.1)
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.show()

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]

    # print(ind)

    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    #
    # #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax =  heatmap.max())
    #plt.colorbar()
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'F:\Lime-pytorch\imagenetresult/' + str(count) + "-2", bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.show()
    #
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2,
                                                hide_rest=False)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask, mode='outer'))
    #plt.show()
    #
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 0, positive_only=False, num_features=5, hide_rest=False)
    # #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.imshow(temp / 2 + 0.5)
    # # plt.axis('off')  # 去坐标轴
    # # plt.xticks([])  # 去刻度
    # # # plt.imshow(img_boundry2)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(r'F:\Lime-pytorch\019result/' + str(count) + "-2", bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.show()

    #print(path)
