# Little programs that Preprocess raw data + Model prediction

Contents: each program function is shown in the file name

**Tips**:
- Quickly open .ipynb file: Open the .ipynb file is very slow in Github and often doesn't load! So when you want to quickly check a .ipynb file from the Github, firstly you can copy the URL of this file. For example there is a .ipynb file in my Github repository, you can easily find this information: https://github.com/GaoBoYu599/Deep-Learning-Applications/blob/master/Unet/Pretreatment/全处理工程：一共4步.ipynb. Secondly, you copy this path into the https://nbviewer.jupyter.org/ , and the .ipynv file can be opened in this website. 

**全过程处理：一共4步.ipynb**:
- Set the parameters for preprocessing, including getting the size of the original image, setting the size of each sub-image, calculating the total number of sub-images
- Cut the original image and save all sub-images
- Turn all images' content into only 0 and 1 values and then resave them: these are the original dataset
- Assemble all the sub-images in the original order into the original image

**多图合并.ipynb**：
- Select some adjacent sub-images for merging. For example, select 4 horizontal and 4 vertical sub-images in the upper left corner for regional splicing and merging.
- This program can be used to 01 sub-images and original 3-channel sub-images
- Save the assembled files in the same file format

**subplot绘图调整.ipynb**:
- Compare and contrast the label sub-images with the original sub-images

**预测后中间层展示.ipynb**: √
- Draw the output of any middle layer(all neurons) during model prediction(01 value images): easily to see what each layer is doing! Very Useful!!
- Check all weights and bias of any middle layer during model prediction(matrix): easily to see the variation of these super-parameters! Very Useful!!

---

# 各种处理原始数据的小程序

内容：各个程序的功能如文件名所示

**注意**:
- 快速打开.ipynb文件：因为在github里直接打开.ipynb文件非常缓慢甚至是加载不出来。所有如果你想快速查看一个github里的.ipynb文件，首先你先找到这个文件的网址（URL），比如在这个仓库下有一个.ipynb文件，可以很容易找到它的URL是：https://github.com/GaoBoYu599/Deep-Learning-Applications/blob/master/Unet/Pretreatment/全处理工程：一共4步.ipynb 。然后，直接拷贝这个URL到https://nbviewer.jupyter.org/ 中，就可以在这个网页下快速查看那个.ipynb文件。

**全过程处理：一共4步.ipynb**:
- 设置预处理的参数，包括获取原始图像的尺寸，设置分割子图的尺寸，计算总子图个数
- 切割原始大图并保存所有子图
- 将所有原始三通道彩色子图转换为只有0和1值的标签图：这些转换后的标签图就是原始的数据集
- 将所有分割后子图按照原始顺序拼合为原来的大图

**多图合并.ipynb**：
- 选取部分彼此相邻的子图进行合并，例如：选取左上角横向4个纵向4个，总计共16个子图进行区域拼接合并。
- 按照相同的文件格式保存区域合并后的图像

**subplot绘图调整.ipynb**:
- 专门负责绘制原始子图和标签子图的对比图

**预测后中间层展示.ipynb**: √
- 绘制模型预测过程中，任意中间层的输出（01值）：很容易查看每个层在做什么！非常有用！
- 查看模型预测过程中，每个中间层的权重和偏置系数（矩阵）：很容易查看这些超参数的变换！非常有用！
