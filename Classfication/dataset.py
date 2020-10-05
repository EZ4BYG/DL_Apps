import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

# 辅助函数1：图像预处理
def load_train(train_path, image_size, classes):
    images = []     # 总图像列表：记录所有读入的，图像的矩阵
    labels = []     # 总标签列表：记录所有读入的，图像的标签
    img_names = []
    cls = []

    print('现在开始读取图像:')
    # 遍历类比：先读island的图像，再读seaice的图像
    for fields in classes:   
        index = classes.index(fields)
        print('现在读取 {} 文件夹下图片 (索引号: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g') # 把文件夹下的所有图片的路径拿到
        files = glob.glob(path) # 遍历每个图片
        # 岛类、海冰的变量
        for fl in files:
            image = cv2.imread(fl)
            # 图像预处理：转换图像成64x64
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            # 归一化：矩阵数值都转成0到1之间
            image = np.multiply(image, 1.0 / 255.0)
            # 加到总图像列表中
            images.append(image)
            
            # 当前图片的标签
            # len(classes)=2，所有label = [0,0]
            label = np.zeros(len(classes))
            # 岛时：label=[1,0]  海冰时：label=[0,1]
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    
    # 把所有列表转换为一维数组/矩阵
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    # 图片已拿到并预处理：把读入并加过标签的数据返回为辅助函数3
    return images, labels, img_names, cls


# 辅助函数2：在迭代过程中起到一些 “记录” 功能
class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  # 被用
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


# 辅助函数3：生成 “训练集” 和 “测试集” 这两个集合！
def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  # 调用辅助函数1：load_train
  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  # 洗牌：打乱原始图片集的图片排列，注意：是整体打乱！标签和图片还是对应的！
  #      整体打乱肯定是在分训练集和测试集之前进行
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  # 设定测试集的样本数：
  if isinstance(validation_size, float):
    # 测试集个数 = 20% * 1400
    validation_size = int(validation_size * images.shape[0])

  # 因为洗牌是在分测试集与训练集之前进行的
  # 所以我取前280个数为测试集！
  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  # 取后1120个为训练集
  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  # 调用函数2：
  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  # 返回一个叫 “data_sets” 的类，该类下有两个集合：“训练集” 和 “测试集”
  return data_sets


