import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
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


inet_model = inc_net.InceptionV3(weights=r"D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
#inet_model.load_weights(r"D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


count = 0
count2 = 0
# for pic_name in os.listdir(r'D:\PyCharm 2021.1.3\pycharmproject\tiny-imagenet-200\tiny-imagenet-200\val\images'):
#     count = 0
#     count2 = count2 + 1
#     folder = os.path.exists(r'D:/PyCharm 2021.1.3/pycharmproject/tiny-imagenet-200/tiny-imagenet-200/val/K=5 N=1000-' + str(count2))
#     if not folder:
#         os.makedirs(r'D:/PyCharm 2021.1.3/pycharmproject/tiny-imagenet-200/tiny-imagenet-200/val/K=5 N=1000-' + str(count2))
#     #img = cv2.imread(r'D:\PyCharm 2021.1.3\pycharmproject\tiny-imagenet-200\tiny-imagenet-200\val\images' + "/" +pic_name, cv2.IMREAD_UNCHANGED)
for count in range(1000):
    count = count + 1
    images = transform_img_fn([os.path.join('data', r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img.png')])
    #image3 = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\superpixel-result4\.jpg')])
    #images = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img.png')])
    image2 = cv2.imread(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\img.png')
    image2 = cv2.resize(image2,(299,299))
    #print(image2.shape)
    #image3 = transform_img_fn([os.path.join('data',r'D:\PyCharm 2021.
    # 1.3\pycharmproject\Lime-pytorch\img3.png')])
    # I'm dividing by 2 and adding 0.5 because of how this Inception represents images

    plt.imshow(images[0] / 2 + 0.5)
    preds = inet_model.predict(images)
    #preds2 = inet_model.predict(image3)
    #print(preds[0,239])
    #print(preds2[0,239])
    # for x in decode_predictions(preds)[0]:
    #     print(x)
    #print(preds)
    #preds_classes = np.argmax(preds)
    #print(preds_classes)
    #创建一个解释器
    explainer = lime_image.LimeImageExplainer()

    #创建结构关系矩阵
    # n = 4
    # block = np.zeros(n)
    # relation = np.zeros((n, n))


    #设置屏蔽颜色
    explanation  = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=5, hide_color=0, num_samples=5000)
    print(explanation.top_labels[0])
    #used_features, neighborhood_data, easymodel = explainer.get_model(images[0].astype('double'), inet_model.predict, num_features=2, top_labels=5, hide_color=0, num_samples=100)
    #print(neighborhood_data[0, [1, 4, 6]])
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
    #easymodel.predict()
    # print(explanation.top_labels)
    # print(explanation.local_exp)
    # print(explanation.segments.shape)
    # print(images.shape)

    # fig = plt.figure('Superpixels')
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)), explanation.segments))
    # plt.axis("off")
    # plt.show()
    # print("------------------------------------------------------------------------------")
    #print(explanation.predict_proba)
    # temp, mask = explanation.get_image_and_mask(285, positive_only=True, num_features=2, hide_rest=True)
    # plt.imshow(images,mask)
    # plt.show()
    #count = 0
    #print(enumerate(np.unique(explanation.segments)))

    # for (i, segVal) in enumerate(np.unique(explanation.segments)):
    #     # print("###################################################################################################################")
    #     # print(explanation.segments)
    #     # print("###################################################################################################################")
    #     #print(segVal)
    #     mask = np.zeros(image2.shape[:2], dtype="uint8")
    #     #mask[:] = 255;
    #     mask[explanation.segments == segVal] = 255
    #     #mask[explanation.segments != segVal] = 255
    #     mask[explanation.segments == segVal + 6] = 255
    #     #mask[explanation.segments == segVal + 2] = 255
    #     #mask[explanation.segments == segVal + 19] = 255
    #     #mask[explanation.segments == segVal + 17] = 255
    #     # mask[segments == segVal + 3] = 255
    #     # mask[segments == segVal + 4] = 255
    #     cv2.imshow("Mask", mask)
    #     #cv2.waitKey(0)
    #     #cv2.imshow("Applied", np.multiply(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
    #     #print(mask.shape)
    #     cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result4/' + str(count) + '.jpg', np.multiply(image2, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) == 0))
    #     #cv2.imwrite('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/superpixel-result2/' + str(count) + '.jpg', np.multiply(image2, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0))
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

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], count2, positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #plt.show()
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去刻度
    # #plt.imshow(img_boundry2)
    # plt.savefig(r'D:/PyCharm 2021.1.3/pycharmproject/tiny-imagenet-200/tiny-imagenet-200/val/K=5 N=1000-'+str(count2)+'/'+str(count)+'.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
    # # img_boundry2.show()
    # # print(3)
    # plt.show()
    #cv2.waitKey(0)

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=4, hide_rest=False)
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.show()

    #temp, mask = explanation.get_image_and_mask(295, positive_only=False, num_features=20, hide_rest=False, min_weight=0.1)
    #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #plt.show()


    #Select the same class explained on the figures above.
    # ind =  explanation.top_labels[0]
    #
    # #print(ind)
    #
    # #Map each explanation weight to the corresponding superpixel
    # dict_heatmap = dict(explanation.local_exp[ind])
    # heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    #
    # #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    # plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    # plt.colorbar()
    # plt.show()
    #
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=True)
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.show()

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=4, hide_rest=False)
    # #plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.imshow(temp / 2 + 0.5)
    # plt.show()
