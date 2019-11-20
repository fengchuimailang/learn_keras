# Sequential 线性的，可以传一堆参数
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 也可以add
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))


# 指定输入形状 shape ，剩下的不需要，因为形状会自己调整
# 三种方法指定shape

# 第一种，input_shape 转给 第一层 tuple（可以有None，代表任意维度）, batch维度并没有指明

# 第二种，dense 应该支持指定输入维度，通过参数input_dim 并且一些 3D temporal layers 支持 input_dim and input_length.

# 指定固定batch size 也可以。 对stateful recurrent networks有用 eg ： batch_size=32 and input_shape=(6, 8)  batch shape (32, 6, 8)

# 下面这两种形式严格相等

# 形式1
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

# 形式2
model = Sequential()
model.add(Dense(32, input_dim=784))

# Compilation 配置
# 第一个参数 optimizer 比如 rmsprop or adagrad 或者 Optimizer子类
# 第二个参数 loss 可以是指定的，也可以是 tensorflow或者Theno的向量
# 第三个参数 metrics 评价标准 可以指定，也可以自己定制

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])


# Train
# keras把numpy arrays 当作数据和标签，使用fit函数适配，callback 在每个stage of the training. 都会被调用


# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

# 多分类任务转为one_hot_label

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)


# Example

