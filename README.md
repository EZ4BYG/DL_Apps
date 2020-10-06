# Deep Learning Examples

Contents: some Deep Learning application examples in earth sciences

# Time: 2018.10.28
- Using Tensorflow1.2 to build CNN network to identify and classify satellite remote sensing images: **Classfication** Folder

**Tips**:
- There are 3 programs in this example, just need to run the train.py
- The dataset using in this example is too large! You can download from https://pan.baidu.com/s/1uSzDadmIjy8V4OuYaC0xGw  password: **xzmd**

# Time: 2020.06.12
- Semantic segmentation of sand ridges based on Unet framework: **Unet** Folder 

**Tips**:
- In the **Unet** folder, there is **Pretreatment** folder which includes many preprocessing functions for raw image data, such as cutting, merging, 0/1 transfer, etc. Please see the readme file in the **Pretreatment** for details
- The data set used in this example comes from real satellite remote sensing images. The whole task includes from image annotation to model training to the final parameters' adjustment, very detailed!
- The image annotation assignment is completed with PS, auxiliary drawing board + some small functions
- All programs are written on jupyter notebook with Python, so all files' suffix is .ipynb

# Time: 2020.10.01
- Neural Network Framework, having the basic functions like Tensorflow, Pytorch: **ANN** folder

**Tips**:
- The framwork has all basic functions: forward propagation, apply gradient descent,  back propagation, predict, etc. 
- Need extra python packages: numpy, matplotlib, sklearn

---

# 深度学习实例：

内容：一些我在实践工作中完成的深度学习在地球科学中的实例

# 时间：2018.10.28
- 利用Tensorflow1.2的深度学习框架搭建CNN网络进行“卫星遥感图像”的分类：**Classfication** 文件夹中

**注意**：
- 一共有3个程序，运行train.py即可；
- 本例用到的数据集太大了！在百度网盘下载：https://pan.baidu.com/s/1uSzDadmIjy8V4OuYaC0xGw  提取码：xzmd 

# 时间：2020.06.12
- Tensorflow2.2框架下，基于Unet神经网络对沙脊线进行高精度语义分割：**Unet** 文件夹中

**注意**：
- 在Unet文件夹中还有一个Pretreatment文件夹，里面是各种对原始图像数据进行预处理的函数：切割、合并、0/1值等；具体细节看Unet文件夹中的说明
- 本例用到的数据集来自真实的卫星遥感影像图片，整个任务包含从图像标注到模型训练好到最后的调参，很详细！
- 标注部分用PS完成，辅助绘图板 + 一个小函数
- 所有的程序都是在jupyter notebook上用python实现的，所有文件后缀都是.ipynb

# 时间：2020.10.01
- 神经网络框架：自己编写的从0到1的神经网络，和Tensorflow和Pytorch拥有一样的基本功能！

**注意**：
- 该网络框架有所有的基础功能：前向传播、应用梯度下降、后向传播、预测等
- 需要额外的python包：numpy、matplotlib、sklearn
    
