from flask import Flask,request,jsonify, make_response,send_file

from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

import os,sys
import base64
import numpy as np

from lime import lime_image

from skimage.segmentation import mark_boundaries
app = Flask(__name__)
interface_url = 'http://127.0.0.1:5000/'


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

@app.route('/', methods=['POST','GET'])
def send_img():
    while True:
        f = request.files['file']
        print('1')

        f.save('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/internaltest/img.png')

        inet_model = inc_net.InceptionV3(
            weights=r"D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
        images = transform_img_fn([os.path.join('data', '/internaltest/img.png')])
        # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
        plt.imshow(images[0] / 2 + 0.5)
        preds = inet_model.predict(images)
        #print(preds)
        #print(decode_predictions(preds)[0])
        # for x in decode_predictions(preds)[0]:
        #     print(x[1])
        #     print(type(x[1]))
        #     #print(np.shape(x))
        #print(type(preds))
        #print(np.shape(preds))
        #print(decode_predictions(preds)[0][0][1])
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict,
                                                            top_labels=5, hide_color=0, num_samples=100)
        # explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=5,
        #                                          hide_color=0, num_samples=100)
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=4,
        #                                             hide_rest=False)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=4, hide_rest=False)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.savefig("D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/internaltest/img2.png")


        #LIME解释功能
        #————————————————————————————————————————————————————————————————————————————————————————————————————————
        #返回图片至固定url
        # files = {'file': ('000.png', open('E:/internaltest/001.jpg', 'rb'), 'file',)}
        # interface = requests.post(interface_url, data={'resourceType': '1', 'resourceTypeExpand': 'pic'},
        #                            files=files)
        # print(interface)
        # print(interface.text)
        #————————————————————————————————————————————————————————————————————————————————————————————————————————
        #————————————————————————————————————————————————————————————————————————————————————————————————————————
        #返回图片二进制
        # with open(r'E:/internaltest/001.jpg', 'rb') as f:
        #     res = base64.b64encode(f.read())
        #     return res
        #————————————————————————————————————————————————————————————————————————————————————————————————————————

        #————————————————————————————————————————————————————————————————————————————————————————————————————————
        #直接返回图片
        #return send_file('D:/PyCharm 2021.1.3/pycharmproject/Lime-pytorch/internaltest/img2.png')
        with open(r'/internaltest/img2.png', 'rb') as f:
            res = base64.b64encode(f.read())
            image=res.decode('ascii')
            out_data = {'code': decode_predictions(preds)[0][0][1], 'data': image}
            return jsonify(out_data)

        #————————————————————————————————————————————————————————————————————————————————————————————————————————


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
