import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as image
import os as os
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import resnet
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.python.keras.optimizer_v2.adam import Adam
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


IMG_SIZE = 224
batch_size = 64
EPOCHS=200
FREEZE_LAYERS=500
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
train_path = '/root/Visual-resnet152/weapons/train2'
val_path='/root/Visual-resnet152/weapons/val'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(
	rescale=1 / 255,
	rotation_range=40,  # 角度值，0-180.表示图像随机旋转的角度范围
	width_shift_range=0.2,  # 平移比例，下同
	height_shift_range=0.2,
	shear_range=0.2,  # 随机错切变换角度
	zoom_range=0.2,  # 随即缩放比例
	horizontal_flip=True,  # 随机将一半图像水平翻转
	validation_split=0.2,
	fill_mode='nearest'  # 填充新创建像素的方法
)

train_generator = train_gen.flow_from_directory(
	directory=train_path,
	shuffle = True,
	batch_size = batch_size,
	class_mode = 'categorical',
	target_size = IMG_SHAPE[:-1],
	color_mode='rgb',
	#classes =classes,
	#subset='training'
)
validation_generator = train_gen.flow_from_directory(
	directory=val_path,
	shuffle = True,
	batch_size = batch_size,
	class_mode = 'categorical',
	target_size =IMG_SHAPE[:-1],
	color_mode='rgb',
	#classes =classes,
	#subset='validation'
 )



model = tf.keras.applications.resnet.ResNet152(input_shape=(224,224,3), weights=None, classes=10)

net_final=model
print('numbers of layers:', len(net_final.layers))



#346  516





filepath='model/resnet152_xin1.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#编译模型
net_final.compile(
optimizer=Adam(lr=1e-5),
	#optimizer='adam',
  loss = 'categorical_crossentropy',
  metrics=['accuracy'])
#打印模型
print(net_final.summary())

#训练模型
history=net_final.fit_generator(train_generator,
steps_per_epoch=max(1, train_generator.n//batch_size),
validation_data=validation_generator,
validation_steps=max(1, validation_generator.n//batch_size),
epochs =200,
#initial_epoch=0,
callbacks=callbacks_list
)
for key in history.history:
	print(key)
#保存模型
#model.save('model/resnet152_model.h5')
#model.load_weights(filepath)
#绘制损失值曲线和准确率曲线
# 记录准确率和损失值
history_dict = history.history
print("history_dict:{}".format(history_dict))
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

# 绘制损失值曲线
plt.figure()
plt.title('resnet')
plt.plot(range(EPOCHS),train_loss,c='k' ,ls='--',label='train_loss')
plt.plot(range(EPOCHS),val_loss,'k' ,label='val_loss' )
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('/root/Visual-resnet152/src/train.png')
plt.savefig('/root/Visual-resnet152/src/val.png')
import matplotlib as mpl
#中文字体设置
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.style"] = "normal"
mpl.rcParams["font.size"] = 10
# 绘制准确率曲线
plt.figure()
#plt.title('InceptionV3-1')
plt.plot(range(EPOCHS), train_accuracy,ls='--', c="k",label="train acc")
plt.plot(range(EPOCHS), val_accuracy,c="k",label="val acc")
plt.ylim(0.5,1)
plt.legend(loc='lower right')
plt.xlabel("train epoch")
plt.ylabel("acc")
plt.savefig('/root/Visual-resnet152/src/train_acc.png')
plt.savefig('/root/Visual-resnet152/src/val_acc.png')
