#!/usr/bin/env python
# coding=utf-8
'''
@描述: 导出模型，并进行测试
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.01.12
@最后编辑人: LiWanglin
@最后编辑时间: 2020.01.12
'''
import tensorflow as tf
from tensorflow import keras

import numpy as np
import cv2

# 读取图片数据
image_6 = cv2.imread("test_image/image6.bmp", cv2.IMREAD_GRAYSCALE)

# 将图片数据进行归一化处理
image_6 = image_6 / 255

# 导出模型
mnist_model = keras.models.load_model("./save_model")

# 使用导出模型，进行测试
a = mnist_model.predict(image_6.reshape(1, 28 ,28))

# 打印预测结果
print(a)
print(np.argmax(a, axis=1))