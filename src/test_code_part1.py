#!/usr/bin/env python
# coding=utf-8
'''
@描述: 训练模型并使用 save_model 保存模型
@版本: V1_0
@作者: LiWanglin
@创建时间: 2020.01.12
@最后编辑人: LiWanglin
@最后编辑时间: 2020.01.12
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

# 打印 Tensorflow 版本以及是否支持 GPU 加速
print("Tensorflow version ：",tf.__version__) 
print(tf.test.is_gpu_available())

if __name__ == "__main__":
    # 导入数据
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    
    train_images = train_images / 255       # 图片数据归一化


    # 添加模型
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # 查看构建模型
    model.summary()

    # 模型编译
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=['acc'])

    # 模型训练
    model.fit(train_images, train_labels, epochs=50)

    # 得到模型准确率
    loss, acc = model.evaluate(test_images, test_labels)
    print("The acc is ",acc*100, "%")

    # 保存模型
    keras.models.save_model(model, "save_model")


