from flask import Flask,request,jsonify, make_response,send_file

from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import tensorflow as tf
import os,sys
import base64
import numpy as np

from lime import lime_image

from skimage.segmentation import mark_boundaries
app = Flask(__name__)
interface_url = 'http://127.0.0.1:5000/'
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        out.append(x)
    return np.vstack(out)

@app.route('/generate', methods=['POST','GET'])
def send_img():
    while True:
        f = request.files['file']
        #print('1')

        f.save('/root/LIME/img.png')
        #f.save(r'F:\Lime-pytorch/img.png')
        inet_model = tf.keras.applications.resnet.ResNet152(include_top=True, input_shape=(224, 224, 3), classes=10,
                                                            weights=None)
        inet_model.load_weights(r'/root/LIME/resnet152_xin1.h5')
        #inet_model.load_weights(r'F:\Lime-pytorch\resnet152_xin1.h5')
        images = transform_img_fn([os.path.join('data', '/root/LIME/img.png')])
        #images = transform_img_fn([os.path.join('data', r'F:\Lime-pytorch/img.png')])
        plt.imshow(images[0] / 2 + 0.5)
        preds = inet_model.predict(images)
        # for x in decode_predictions(preds)[0]:
        #     print(x)
        explainer = lime_image.LimeImageExplainer()
        mean_value = np.mean(images[0])
        explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=2, hide_color=mean_value, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask, mode='outer'))
        plt.savefig("/root/LIME/img2.png")
        #plt.savefig(r"F:\Lime-pytorch/img2.png")
        if preds[0].any() == 0:
            id = "手枪"
        elif preds[0].any() == 1:
            id = "突击步枪"
        elif preds[0].any() == 2:
            id = "直升机"
        elif preds[0].any() == 3:
            id = "榴弹炮"
        elif preds[0].any() == 4:
            id = "客机"
        elif preds[0].any() == 5:
            id = "两栖攻击舰"
        elif preds[0].any() == 6:
            id = "狙击步枪"
        elif preds[0].any() == 7:
            id = "主战坦克"
        elif preds[0].any() == 8:
            id = "火箭"
        else:
            id = "J系列战斗机"
        with open(r"/root/LIME/img2.png", 'rb') as f:
        #with open(r"F:\Lime-pytorch/img2.png", 'rb') as f:
            res = base64.b64encode(f.read())
            image=res.decode('ascii')
            out_data = {'ID': id, 'data': image}
            return jsonify(out_data)

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
