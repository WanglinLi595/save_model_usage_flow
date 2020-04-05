#!/usr/bin/env python
# coding=utf-8
'''
@描述: 
@版本: V1_0
@作者: LiWanglin
@创建时间: Do not edit
@最后编辑人: LiWanglin
@最后编辑时间: Do not Edit
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def preprocess(x, y):
    # [0-1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def main():
    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y, axis=1)
    print(x.shape, y_shape, x_test.shape, y_test.shape)
    # 构建训练集对象，随机打乱
    train_bd = tf.data.Dataset.from_tensor_slices((x, y))
    train_bd = train_bd.shuffle(1000).map(preprocess).batch(128)
    # 构建测试集对象
    test_bd = tf.data.Dataset.from_tensor_slices((x, y))
    test_bd = train_bd.shuffle(1000).batch(128)
    # 
    sample = next(iter(train_bd))
    print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

    conv_layers = [
        # 卷积层
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation='relu'),
        # 卷积层
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation='relu'),
        # 池化层
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # 卷积层
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation='relu'),
        # 卷积层
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation='relu'),
        # 池化层
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # 卷积层
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation='relu'),
        # 卷积层
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation='relu'),
        # 池化层
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # 卷积层
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation='relu'),
        # 卷积层
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation='relu'),
        # 池化层
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # 卷积层
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation='relu'),
        # 卷积层
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation='relu'),
        # 池化层
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
    ]

    conv_net = keras.Sequential(conv_layers)

    # 创建 3 个全连接层
    fc_net = keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation=None)
    ])

    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])

    conv_net.summary()
    fc_net.summary()

    variadles = conv_net.trainable_variables + fc_net.trainable_variables

    grasd = keras.tape.gradinet(loss, variadles)
    keras.optimizers.apply_gradients(zip(grads, variadles))





if __name__ == "__main__":
    main()