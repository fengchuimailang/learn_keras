
# 这是一些常见问题在keras中


# 引用
# @misc{chollet2015keras,
#   title={Keras},
#   author={Chollet, Fran\c{c}ois and others},
#   year={2015},
#   howpublished={\url{https://keras.io}},
# }

# 在GPU上面跑
# 如果是Tensorflow或者CNTK后端 ，自动检测GPU
# Theano 三种方法
# 方法1 flags 也就是环境变量
# THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py

# 方法2 配置        .theanorc
# 方法3 配置 theano.config.device 和 theano.config.floatX 在文件的开始

import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'

# 多GPU
# 两种方法 
# 方法1 数据并行
# 方法2 设备并行
# 多数 多数的时候是数据并行
# 数据并行 每一个设备都有一分模型的拷贝 使用keras.utils.multi_gpu_model 最快到8倍GPU

# example
from keras.utils import multi_gpu_model

x = None
y = None
# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')
# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)




# 设备并行 在不同模型上运行同一模型的不同部分
# 最好是有并行部分的模型 比如有两个分支的模型

# shared LSTM模型在两个不同的序列模型上并行
import keras
import tensorflow as tf

# Model where a shared LSTM is used to encode two different sequences in parallel
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# Process the first sequence on one GPU
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# Process the next sequence on another GPU
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# Concatenate results on CPU
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)


# sample batch epoch的概念
# Sample : 数据集只能够的一个元素
# Batch : N个sample构成的集合
#  在sample中的batch分别进行处理 
# 每个batch将更新模型参数一次
# inference的batch尽可能的大，因为速度快，但是太大会可能outofmamory
# epoch是一个随意的截断，通常定义为整个数据集遍历一遍
# epoch 把训练分成不同的阶段 对于记载日志和周期性评估很有好处
# 使用validation_data和validation_split的时候，评估会在每一个epoch之后
# 可以添加callback 在每一个epoch之后 ，可以改变学习率或者保存模型

# 怎样保存keras模型
# 保存/加载 整个模型 (architecture + weights + optimizer state)
# 不推荐使用pickle或者cpickle来保存keras模型

# 可以使用model.save(filepath) 来把keras模型保存在HDF5中

# 模型包含以下四点
# the architecture of the model, allowing to re-create the model
# the weights of the model
# the training configuration (loss, optimizer)
# the state of the optimizer, allowing to resume training exactly where you left off.


# 可以使用 keras.models.load_model(filepath) 来加载模型
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')


# 保存/加载 模型结构 而不保存权重和训练配置

# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
# 生成的JSON/YAML文件是可读的

# 可以从这些文件中创建一个模型
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)

# Saving/loading only a model's weights
# 保存模型权重可以用HDF5
model.save_weights('my_model_weights.h5')

# 如果已经有代码可以实例化模型，加载权重
model.load_weights('my_model_weights.h5')

# 如果要加载权重到不同的模型（有一些层次是相同的）比如要调优或迁移学习，可以by name

model.load_weights('my_model_weights.h5', by_name=True)

# example

"""
Assuming the original model looks like this:
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)


# Handling custom layers (or other custom objects) in saved models
# 如果加载的模型有一些自己的类或者功能性的类别

# 也就是说加载某一层

from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})


# custom object scope
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')

# 加载个性化模型和加整个模型的方法一样 
# load_model, model_from_json, model_from_yaml

# example
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})

# 为什么训练集的损失比测试集合损失还要高

# keras模型有两个模型：训练模型、测试模型
# 在测试阶段 没有Dropout或者是L1和L2损失

# 除此之外，训练集的是训练集每一个batch的平均


# 获取中间层的输出

# 创建一个你最喜欢的模型

from keras.models import Model

model = ...  # create the original model
data = None

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)


# 也可以创建一个Keras函数给定特定输入，输出特定输出
# example
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]


# 如果因为 Dropout, BatchNormalization, etc 在训练和测试阶段有不同的表现、
# 你需要传一个学习

#  learning_phase中1是 train 0 是test
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]


# 数据可以不全在内存空间
# 可以 model.train_on_batch(x, y) 和 model.test_on_batch(x, y)

# 或者是 model.fit_generator(data_generator, steps_per_epoch, epochs).




# 如果验证集损失不下降，可以停止训练 用callback

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # 就是停止条件
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])


# 验证集怎样分割

# 设置 validation_split 参数 在model.fit中, 如果是 0.1 就取得最后的10%的数据
# 如果0.25 取最后25% 数据
# 注意，在分离验证集之前 数据不会shuffled 所以验证集仅仅是最后的 10%


# 在训练的过程中，数据是shuffled的吗
# 如果 shuffle argument in model.fit is set to True
# 训练数据会shuffle

# 验证集也不需要shuffled

# 怎样记录 训练损失 训练精度 验证损失 验证精度等等 在每一个epoch

hist = model.fit(x, y, validation_split=0.2)
print(hist.history)


# 怎样freeze keras的层次

# freeze 意味着它竟被从训练集中剔除出去。他的权重不会被更新
# 这对fine-tuning很有用处，或者使用固定的Embedding在 文本的输入

# 你可以传入一个参数argument到一个层次构造器 来使得它更不可训练
from keras.layers import Dense
frozen_layer = Dense(32, trainable=False)
# 也可以在实例化之后设置 trainable 属性 compile() 最终保证生效


# 可以怎样使用stateful RNNs
# 这意味着 每一个batch的 样本的状态会最为下一个样本的初始状态

# 使用 stateful RNNs 基于假设
# 假设1 所有的batches有同样数量的样本
# 假设2 有连续的batches x1 和x2 
# x2[i] is the follow-up sequence to x1[i]

# 使用 statefulness 需要
# 1. 精准确定batch size
# 2. 设置 stateful=True 对于RNN layers
# 3. 指定shuffle=False

# reset the states accumulated
# model.reset_states() resent所有状态
# layer.reset_states() reset某一层的状态

# Example
x # this is our input data, of shape (32, 21, 16)

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()


# How can I remove a layer from a Sequential model?
# 移除最后一层用 .pop()

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"

# 怎样使用预训练模型

# 分类模型
# Xception
# VGG16
# VGG19
# ResNet
# ResNet v2
# ResNeXt
# Inception v3
# Inception-ResNet v2
# MobileNet v1
# MobileNet v2
# DenseNet
# NASNet


# 可以从keras.applications 中获取

# 怎样使用HDF5作为keras输入
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)


# keras配置文件 在  $HOME/.keras/
# windows 用户 replace $HOME with %USERPROFILE%.
# keras配置是一个JSON文件 存储在$HOME/.keras/keras.json
# 默认配置是

# {
#     "image_data_format": "channels_last",
#     "epsilon": 1e-07,
#     "floatx": "float32",
#     "backend": "tensorflow"
# }

# 理解image_data_format
# 128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）

# 第一 image data format 默认的通道的位置
# 第二 epsilon 防止除0错误
# 第三 默认的浮点数据类型
# 第四 默认后端 

# 缓存的dataset文件 比如用get_file下载的 被存储在 $HOME/.keras/datasets/.

# 我在训练中要怎样使用 keras的可复制结果

# 首先设置 PYTHONHASHSEED 环境变量为0 在程序的开始

# 一个在python中设置环境变量的方法

#  cat test_hash.py
# print(hash("keras"))
# $ python3 test_hash.py                  # non-reproducible hash (Python 3.2.3+)
# -8127205062320133199
# $ python3 test_hash.py                  # non-reproducible hash (Python 3.2.3+)
# 3204480642156461591
# $ PYTHONHASHSEED=0 python3 test_hash.py # reproducible hash
# 4883664951434749476
# $ PYTHONHASHSEED=0 python3 test_hash.py # reproducible hash
# 4883664951434749476


#当运行Tensorflow后端GPU的时候，一些操作的结果是不固定的，
# 因为GPU并行计算，完成时间是不确定 所以执行结果的顺序不总是保证的
# 浮点数的确定行，几个数，顺序不一样，结果也可能不一样
# 最简单的方式是在CPU上运行
# CUDA_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py

# 下面的片段解释了怎样获得可复制的结果
# 环境是 TensorFlow backend for a Python 3 environment

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...

# 安装 HDF5 或者 h5py 来保存代码到keras

# sudo apt-get install libhdf5-serial-dev

