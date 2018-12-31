import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
# conda install --channel https://conda.anaconda.org/menpo opencv3
#Adding Seed so that random initialization is consistent

# 随机初始化：每一次随机都是相同的结果
from numpy.random import seed
seed(10) # 数值随意，不改变就行
from tensorflow import set_random_seed
set_random_seed(20)


# 每次迭代送入32张图
batch_size = 28
# 指定标签：只有两种
classes = ['island','seaice']
# 一共有多少类别，最后训练需要
num_classes = len(classes)


# 1000张图中，20%为测试集
validation_size = 0.2
# 由于图片有大有小，这里我设置/截所有图像为64x64
img_size = 64
# 彩色图：3通量
num_channels = 3
# 所有图像所在位置：training_data文件夹
train_path='training_data'

# 调用dataset.py中的read_train_sets函数
# 目的：得到已处理好、分好的图片集
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("图片预处理完毕，分组如下：")
print("训练集张数:{}".format(len(data.train.labels)))
print("测试集张数:{}".format(len(data.valid.labels)))



session = tf.Session()
# 第一个参数是每次送入训练的张数；中间两个参数是图片的尺度：64x64；最后一个参数是3通量
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# 标签数
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
# argmax判断当前哪个值大，就是什么
y_true_cls = tf.argmax(y_true, dimension=1)



# 卷积网络结构设置
filter_size_conv1 = 3    # 卷积核1是3x3
num_filters_conv1 = 32   

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
# 全连接层：1024个特征  常见还有2048 4096等等
fc_layer_size = 1024

# 权重参数设置
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

# -------------------------------------------------------- #
# 辅助函数1：卷积池化各层参数的设置与工作
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    
    # 权重参数初始化：3 3 3 32
    # 前两个参数是滑动窗尺寸3x3 第三个参数为3通量 第4个参数是这一层观察多少个特征
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    # 执行一次卷积操作
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases
    
    # relu激活函数：激活后的数值传给下一步
    layer = tf.nn.relu(layer)
    
    # 池化操作：上一卷积层的输出就是这里的输入；池化窗尺寸为2x2
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    # 一套卷积池化层工作完毕
    return layer

# 辅助函数2：全连接层参数的设置
def create_flatten_layer(layer):
    # 把到最后一个卷积池化层的尺寸：[7,8,8,64]
    # 既然是全连接层，就要把它拉伸；即对这个全连接层的"输入"应有8x8x64=4096个
    layer_shape = layer.get_shape()

    # 计算得到那个4096的数值
    num_features = layer_shape[1:4].num_elements()

    # 转换成数组/矩阵
    layer = tf.reshape(layer, [-1, num_features])

    return layer

# 辅助函数3：全连接层创建与工作
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    # 全连接层的权重参数等的设置：前4096，后2048（自己之前设置过的）
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # 计算：
    layer = tf.matmul(input, weights) + biases
    
    # dropout解决过拟合，随机杀死30%个节点
    layer=tf.nn.dropout(layer,keep_prob=0.7)
    
    # relu激活函数
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer
# -------------------------------------------------------- #
    

# 调用辅助函数1：第一套 “卷积池化层” 工作完毕
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

# 调用辅助函数1：第二套 “卷积池化层” 工作完毕
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

# 调用辅助函数1：第三套 “卷积池化层” 工作完毕
layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
         
# 调用辅助函数2：对下面的全连接层先设置相关的参数
layer_flat = create_flatten_layer(layer_conv3)

# 调用辅助函数3：全连接层1工作
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

# 调用辅助函数3：全连接层2工作
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

# 预测值：谁大选谁
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

# 以下是常规操作：
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer()) 

# 打印：
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- 训练集精度: {2:>6.1%}, 测试集精度: {3:>6.1%},  测试集损失值: {4:.3f}"
    print(msg.format(epoch + 1,i, acc, val_acc, val_loss))
    ftmp.write(msg.format(epoch + 1,i, acc, val_acc, val_loss))
    ftmp.write('\n')
    ftmp.flush()

# 从第0次开始迭代
total_iterations = 0


# 跑模型 + 模型保存：
ftmp = open('result.txt','w')
saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        # 调用dataset.py文件中data类中的next_batch方法
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        # 做好每一个epoch
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        # 开始运行训练
        session.run(optimizer, feed_dict=feed_dict_tr)
        # 打印：
        if i % int(data.train.num_examples/batch_size) == 0: 
            # 损失值
            val_loss = session.run(cost, feed_dict=feed_dict_val)  
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,i)
            # 保存文件夹的地方
            saver.save(session, './model/island-seaice.ckpt',global_step=i) 


    total_iterations += num_iteration

# 调用train函数：规定一共迭代4000次
train(num_iteration=4000)
