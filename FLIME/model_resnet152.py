#!usr/bin/env python
# _*_ coding:utf-8 _*_

# 作者： 刘璐
# 日期：2022/6/26
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend
from gradcam import GradCAM
class Resnet152Wrapper_public():
  def __init__(self, include_top, weights, labels_path, gradcam_layer=None):
      self.model = tf.keras.applications.resnet.ResNet152(include_top=include_top, weights=weights,input_shape=(224,224,3),classes=10)
      if not gradcam_layer is None:
        self.gradcam_layer = gradcam_layer
      else:
          self.gradcam_layer = self.find_target_layer()
      self.layers = self.model.layers
      self.sess_array = backend.get_session()
      # print("self.model.weights is {}".format(self.model.weights))
      # print(len(self.model.weights))
      # print(11111111111111111)

      self.w = self.sess_array.run(self.model.weights[930])
      self.b = self.sess_array.run(self.model.weights[931])
      # print("self.w is {}".format(self.w))
      # print("self.b is {}".format(self.b))
      # print(1111111111111111111111111111111111111111111111111111111111111111111)

      self.find_target_layer_idx()

      GradcamModel = Model(
          inputs=[self.model.inputs],
          outputs=[self.model.get_layer(self.gradcam_layer).output, self.model.output]
      )
      self.gradCAM_model = GradCAM(self.model, self.gradcam_layer, GradcamModel, self.sess_array)


      with open(labels_path, 'r') as f:
          self.labels = json.load(f)



  def get_image_shape(self):
      return (224,224)


  def run_examples(self, images, BOTTLENECK_LAYER):
      new_model = Model(
          inputs=[self.model.inputs],
          outputs=[self.model.get_layer(BOTTLENECK_LAYER).output, self.model.output]
      )
      x = (images * 255).copy()
      x = tf.cast(preprocess_input(x), tf.float32)
      (LayerOuts, preds) = new_model(x)
      return self.sess_array.run(LayerOuts)


  def label_to_id(self, CLASS_NAME):
      return int(self.labels[CLASS_NAME.replace(' ', '_')])


  def get_gradient(self, activations, CLASS_ID, BOTTLENECK_LAYER, x):
      gradModel = Model(
          inputs=[self.model.inputs],
          outputs=[self.model.get_layer(BOTTLENECK_LAYER).output, self.model.output]
      )
      inputs = tf.cast(np.expand_dims(x, axis=0), tf.float32)
      (convOuts, preds) = gradModel(inputs)  # preds after softmax
      loss = preds[:, CLASS_ID[0]]
      grads = tf.gradients(loss, convOuts)
      return -1*self.sess_array.run(grads)[0]


  def find_target_layer(self):
      for layer in reversed(self.model.layers):
      # for layer in (self.model.layers):
          print(layer)
          print(layer.name)
          print(11)
          if 'conv' in layer.name:
              return layer.name
      raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")


  def find_target_layer_idx(self):
      self.target_layer_idx = {}
      for idx, layer in enumerate(self.model.layers):
          self.target_layer_idx[layer.name] = idx


  def get_linears(self, x):
      pool_value = self.run_examples(x, 'avg_pool')
      # print("self.b is {}".format(self.b))
      # print("self.w is {}".format(self.w))
      # print("pool_value.dot(self.w) is {}".format(pool_value.dot(self.w)))
      # print("len b is {}".format(len(self.b)))
      # print("len w is {}".format(len(self.w)))
      # print(type(self.b))
      # print(type(self.w))
      # print("len dot w is {}".format(len(pool_value.dot(self.w))))
      #w=
      #self.b =np.ones(7,dtype=float)
      #pool_value.dot(self.w) is [-212.32944 - 267.12842 - 214.88748 - 217.7695 - 228.1886]
      #res=[-212.32944 - 267.12842 - 214.88748 - 217.7695 - 228.1886]+self.b
      #print(len(res))
      #print(res)
      res = pool_value.dot(self.w) + self.b
      print(res)
      return res