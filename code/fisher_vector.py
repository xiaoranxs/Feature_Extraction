# author: xiao ran
# time: 2023-7-25
# using: 对类别视频进行FV编码，并保存

import os
import traceback
from random import sample
import requests
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize


def send_message(title, content, SCKEY):
    url = f'https://sctapi.ftqq.com/{SCKEY}.send'
    data = {
        'title': title,
        'desp': content
    }
    response = requests.post(url, data=data)
    return response.json()

# 提取图像的光流特征
def extract_features(frame, prev_gray):
    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        # 如果是第一帧，则只更新 prev_gray 并返回空特征
        prev_gray = gray
        return None, prev_gray

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 将光流向量2D转换为特征向量1D表示
    features = flow.flatten()
    # 更新前一个帧
    prev_gray = gray

    return features, prev_gray

# 对视频循环读取图像，提取图像的光流特征，所有特征合成矩阵，并进行PCA降维和标准化
def extract_video_features(video_path, dim):
    # 循环对视频进行截取
    cap = cv2.VideoCapture(video_path)
    features = []
    prev_gray = None  # 设置初始的前一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        feature_vector, prev_gray = extract_features(frame, prev_gray)
        features.append(feature_vector)
    cap.release()

    # 消去None
    features[:] = [x for x in features if x is not None]
    # 将特征进行向量化和规范化
    features = np.array(features)

    # 进行特征向量的降维，以便通过PCA
    if dim - features.shape[0] + 1 > 0:  # 样本数量不足以进行pca，随机选取行复制
        add_row = dim - features.shape[0] + 1
        index = sample(list(range(0, features.shape[0])), add_row)
        new_rows = []
        for i in index:
            new_rows.append(features[i, :])
        new_rows = np.array(new_rows)
        features = np.vstack((features, new_rows))

    pca = PCA(n_components=dim)
    features = pca.fit_transform(features)

    # 规范化特征向量
    features = normalize(features)

    return features

def encode_fisher_vectors(features, K):
    # 使用高斯混合模型对特征进行编码
    gmm = GaussianMixture(n_components=K, covariance_type='diag')  # K为高斯混合模型中的分量数量
    gmm.fit(features)
    encoded_vectors = gmm.predict_proba(features)

    # 进行Fisher向量编码
    fisher_vectors = []
    for response in encoded_vectors:
        response = response[:, np.newaxis]  # 给response增加新的维度，将其形状变为（10，1）
        fv = np.concatenate([np.sum(response * gmm.means_, axis=0),
                             np.sum(response * (gmm.covariances_ + gmm.means_ ** 2), axis=0)])
        fisher_vectors.append(fv)

    # 将所有帧的Fisher向量编码汇总为视频的Fisher向量编码
    video_fv = np.sum(fisher_vectors, axis=0)
    # 可选的后处理，例如规范化Fisher向量编码
    video_fv = normalize(video_fv.reshape(1,-1))

    return video_fv

def category_fvcoding(category, K, dim):
    coding_path = "./Video_Coding/" + category  # 存放类别视频编码的路径
    folder_path = "./Video/" + category  # 类别视频的文件夹路径
    pathDir = os.listdir(folder_path)  # 获取文件夹下的视频名称

    category_coding = []  # 类别编码
    num = 0
    for video_path in pathDir:
        num += 1
        print("Video #" + str(num))
        features = extract_video_features(folder_path + "/" + video_path, dim)
        category_coding.append(encode_fisher_vectors(features, K))

    category_coding = np.squeeze(np.array(category_coding))  # 多余维度消去
    np.save(coding_path + ".npy", category_coding)


if __name__ == "__main__":
    # 替换为你的SCKEY
    SCKEY = 'SCT217619Tu1XH2CF1ry2KDdER3N0AUQcL'

    folder_path = "./Video" # 类别视频的文件夹路径
    pathDir = os.listdir(folder_path)  # 获取文件夹下的视频名称
    # 过滤出目录中的文件夹
    folders = [item for item in pathDir if os.path.isdir(os.path.join(folder_path, item))]

    error_list = []
    for i in folders:
        if os.path.isfile("./Video_Coding/" + i + ".npy"):
            continue  # 已经计算，则跳过

        try:
            print("Category # " + i)
            category_fvcoding(i, 20, 20)
        except Exception as e:
            error_list.append(i)

    print(e)
    # 发送消息
    title = 'bug'
    content = f'程序出现异常：{str(e)}\n\n{traceback.format_exc()}'
    response = send_message(title, content, SCKEY)
    print(response)



