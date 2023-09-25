from flask import Flask,request,jsonify, make_response,send_file

from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

import os,sys
import json
import numpy as np
import requests
from lime import lime_image

from skimage.segmentation import mark_boundaries
app = Flask(__name__)
interface_url = 'http://127.0.0.1:12001/'
url = 'http://127.0.0.1:8080/kjs/fake/fakeTask/acceptFakeTaskResult'
#url = 'http://10.132.100.45:8080/kjs/fake/fakeTask/acceptFakeTaskResult'


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
        data1 = request.get_json()
        #print(data1)
        #print(data1.type())
        #f.save(r'/root/data1/10361/Lime-pytorch/interface/img.png')
        taskId = data1['taskId']
        timestamp = data1['timestamp']
        #print(taskId)
        #print(timestamp)
        #print('1')

        #f.save(r'/root/data1/10361/Lime-pytorch/interface/img.png')

        inet_model = inc_net.InceptionV3(
            weights=r"/root/LIME/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
        images = transform_img_fn([os.path.join('data', r'/root/LIME/img.png')])
        # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
        plt.imshow(images[0] / 2 + 0.5)
        preds = inet_model.predict(images)
        for x in decode_predictions(preds)[0]:
            print(x)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=3, hide_color=0, num_samples=100)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 0, positive_only=True, num_features=5, hide_rest=True)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.savefig(r"/root/LIME/img2.png")
        # print("***************************************************************************************")
        # print("***************************************************************************************")
        # print("***************************************************************************************")

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
        #return send_file(r'/root/data1/10361/Lime-pytorch/interface/result.png')
        #data = request.get_json()
        data2 = {"taskId": taskId, "result": {"uploadUrl": r'/root/LIME/img2.png', "list": [], "json": {}},"timestamp": timestamp}
        requests.post(url, data=json.dumps(data2), headers={'Content-Type': 'application/json'})
        return {"code":200,"result":[],"success":True,"message":"发送成功","timestamp":timestamp}
        #————————————————————————————————————————————————————————————————————————————————————————————————————————


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=12001)
