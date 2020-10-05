# Self-written Neural Network framework


This ANN has all basic functions, such as forward propagation, apply gradient descent,  back propagation, predict, etc. Just need 3 extra basic python packages. Now let's explain what each file does. What's more, if you want to use a better optimization algorithm, like Adam, you just need to change the contents of the *nn_applygradient.py* file. Therefore, this framework has a good expansion! 

**Tips**:
- test.py: main function, run this file. Various parameter adjustments are also carried in it
- nn_train.py: responsible for training, it will call *nn_forward.py*, *nn_applygradient.py* and *nn_backpropagation.py*
- nn_forward.py: responsible for forward calculation, calculating predictive value
- nn_applygradient.py: gradient descent is performed on the objective function, finding the direction of model parameter optimization
- nn_backpropagation.py: change each weight parameter from back to front
- nn_test.py: responsible for testing the effect of the current model on the validation set
- nn_predict.py: responsible for testing the effect of the final model on the test set

---

# 自己编写的神经网络框架

这个ANN框架包含各种基本功能，例如前向传播、反向传播、应用梯度下降优化算法等。仅需要额外3个基本的Python包即可。现在解释每个文件的作用。此外，如果你想使用更好的目标函数优化算法，例如Adam，你需要改变*nn_applygradient.py*文件里的内容即可！因此，该框架具有很好的拓展性！

**注意**：
- test.py：主函数，运行该文件。各种参数调整也是在该函数内修改即可
- nn_train.py：主要负责训练部分的函数，它会自动调用*nn_forward.py*，*nn_applygradient.py*，*nn_backpropagation.py*功能函数
- nn_forward.py：前向计算函数，计算预测值
- nn_applygradient.py：对目标函数进行梯度下降优化，找到模型参数优化的方向
- nn_backpropagation.py：根据梯度下降提供的信息，从后往前对目标函数内每个权重参数进行优化
- nn_test.py：当前模型在验证集上的效果
- nn_predict.py：最终模型在测试集上的效果
