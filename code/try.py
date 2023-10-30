# author:
# time:
# using:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载示例数据集（鸢尾花数据集）
iris = load_iris()
X = iris.data
y = iris.target

# 将目标变量进行one-hot编码
y = to_categorical(y)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 构建神经网络模型
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
model.add(Dense(3, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("准确率：", accuracy)
