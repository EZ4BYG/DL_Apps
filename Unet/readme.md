![image](https://github.com/EZ4BYG/DL_Apps/blob/master/Unet/Result.jpg)

**Tips**:
- Ordinary files: They are all Unet related programs(different network structure), you can just try all of that or just run the 沙漠Unet_8x8.ipynb
- Pretreatment folder: There are many preprocessing functions
- Dataset folder: There are 450 images(256x256) that are already labeled. There are only 0 and 1 values in each image!
- Model folder: A trained model that already performs well on the validation set(Accuracy rate above 97%)! It also satisfies the precision requirement on the test set.
- All .ipynb programs need to be run on GPU-version Tensorflow2.x! Programs can use more than one GPU at a time

---

**注意**：
- 普通文件：他们都是和Unet网络训练相关的程序（网络结构不同），都可以运行或者就运行 **沙漠Unet_8x8.ipynb** 这个文件即可
- Pretreatment文件夹：里面都是预处理相关的函数，具体内容看其中的readme.md文件
- Dataset文件夹：理由有450张已经坐标标注的原始图像（256x256），也就是原始数据。每张图片里都只有0和1值！
- Model文件夹：已经已经训练好的模型，它在验证集上已经有97%以上的准确度！并且它在测试集上也已经满足精度需求
- 所有.ipynb文件夹都需要运行在GPU版本的Tensorflow2.0以上版本。每个程序都可以同时调用多张显卡
