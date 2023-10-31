# author: xiao ran
# time: 2023-7-29
# using: 将类别编码的npy文件转存为csv文件，以便R分类

import numpy as np
import csv
import os

# npy文件转为csv文件
def npy_to_csv(category):
    data = np.load('./Video_Coding/' + category + '.npy')

    with open('./Video_Coding/' + category + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# 设置文件夹路径
folder_path = "./Video_Coding"
# 获取文件夹下所有 .npy 文件的文件名
npy_files = [file for file in os.listdir(folder_path) if file.endswith(".npy")]

for i in npy_files:
    if os.path.exists('./Video_Coding/' + i[:-4] + '.csv'):
        continue

    npy_to_csv(i[:-4])

