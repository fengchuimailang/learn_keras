# simplest type of model is the Sequential model
from keras.models import Sequential
model = Sequential()

# add
from keras.layers import Dense
model.add(Dense(units=64,activation="relu",input_dim=100))
model.add(Dense(units=10,activation="softmax"))

# configure its learning process with .compile()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# https://blog.csdn.net/legalhighhigh/article/details/81348879
# 一句话解释就是连续计算两次梯度
import keras
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

# 迭代训练数据
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
x_train = None
y_train = None
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 也可以一步一步更新权值
x_batch = None
y_batch = None
model.train_on_batch(x_batch, y_batch) # 这个是一个batch


# 评估模型
x_test = None
y_test = None
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


# 新数据集上预测
classes = model.predict(x_test, batch_size=128)

# 


