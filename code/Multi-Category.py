# author: xiao ran
# time: 2023-7-27
# using: 对fv进行多分类

import os
from random import sample
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 进行多分类
def multi_category(n, category_list=None):
    """
    :param n: 欲进行多分类的类别数
    :param category_list: 若n=0，则对category_list中特定类别进行分类
    :return: 准确率
    """
    # 1.输入处理
    if n == 0 and category_list is None:
        print("参数错误")
        return
    elif n != 0 and category_list is not None:
        print("不能同时指定n和category_list")
        return
    elif n != 0:  # 随机抽取类别进行多分类
        pathDir = os.listdir('./Video_Coding')
        category_list = sample(pathDir, n)

    # 2. 构造dataset
    X = np.load('./Video_Coding/' + category_list[0])
    y = np.array([0]*X.shape[0])
    for i in list(range(len(category_list)))[1:]:
        cate_arr = np.load('./Video_Coding/' + category_list[i])
        X = np.vstack((X, cate_arr))
        y = np.concatenate((y, [i]*cate_arr.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    y_NN = to_categorical(y)
    X_train_NN, X_test_NN, y_train_NN, y_test_NN = train_test_split(X, y_NN, test_size=0.3, random_state=20)

    # 3. 分类
    # 创建分类器
    rf_classifier = RandomForestClassifier(n_estimators=100)  # 随机森林
    svm_classifier = SVC()  # SMV
    model = Sequential()  # 构建神经网络模型
    model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(y_NN.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    rf_classifier.fit(X_train, y_train)
    svm_classifier.fit(X_train, y_train)
    model.fit(X_train_NN, y_train_NN, epochs=50, batch_size=10, verbose=1)

    # 在测试集上进行预测
    y_pred1 = rf_classifier.predict(X_test)
    y_pred2 = svm_classifier.predict(X_test)

    # 计算准确率
    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy2 = accuracy_score(y_test, y_pred2)
    loss, accuracy3 = model.evaluate(X_test_NN, y_test_NN, verbose=0)

    print("random forest: " + str(accuracy1) + "\nSVM: " + str(accuracy2)
          + "\nNN: " + str(accuracy3) + "  loss: " + str(loss))


multi_category(10)

