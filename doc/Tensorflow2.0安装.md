<!--
 * @描述: 
 * @版本: V1_0
 * @作者: LiWanglin
 * @创建时间: 2020.01.11
 * @最后编辑人: LiWanglin
 * @最后编辑时间: 2020.01.15
 -->

# Tensorflow2.0安装

## 一. 安装 CPU 版

- 输入命令：conda create -n Tensorflow2.0_CPU python=3.6，然后输入 y ，创建新的虚拟环境。
- 安装完成后，输入：conda env list
可以看到 Tensorflow2.0_CPU 虚拟环境安装成功

    <div align=center>
    <img src=https://github.com/WanglinLi595/Save_Markdown_Picture/blob/master/tensorflow_usage_flow/tensorflow_cpu_2.png?raw=true>
    </div>

- 激活 Tensorflow2.0_CPU 虚拟环境：
activate Tensorflow2.0_CPU
- 输入 conda install tensorflow=2.0.0，安装 tensorflow 2.0 版本
    <div align=center>
    <img src=https://github.com/WanglinLi595/Save_Markdown_Picture/blob/master/tensorflow_usage_flow/tensorflow2.0_cpu_3.png?raw=true>
    </div>

## 二. 安装 GPU 版

- 输入命令：conda create -n Tensorflow2.0_GPU python=3.6，然后输入 y ，创建新的虚拟环境。
- 激活 Tensorflow2.0_GPU 虚拟环境：
activate Tensorflow2.0_GPU
- 输入 conda install tensorflow-gpu==2.0.0 ，安装 tensorflow 2.0 GPU 版本
- 安装完成后，在 vscode 里面切换虚拟环境为Tensorflow2.0_GPU，输入代码：

    ```python
    import tensorflow as tf

    print('-'*10,"\n")
    print(tf.test.is_gpu_available())
    print('-'*10,"\n")
    ```

如果打印结果为：Ture，则 tensorflow GPU 版安装成功。
    <div align=center>
    <img src=https://github.com/WanglinLi595/Save_Markdown_Picture/blob/master/tensorflow_usage_flow/tensorflow_GPU_1.png?raw=true>
    </div>