from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import *

from Class import *
from nn_train import nn_train
from nn_forward import nn_forward
from nn_test import nn_test
from nn_predict import nn_predict
from nn_backpropagation import nn_backpropagation
from nn_applygradient import nn_applygradient
from function import sigmoid, softmax

import numpy as np


def nn_testChess():
    with open('krkopt.data') as my_data:  # 读取文件部分
        lines = my_data.readlines()
        data = np.zeros((28056, 6), dtype=float)
        label = np.zeros((28056, 2), dtype=float)
        i = 0
        for line in lines:
            line = line.split(',')  # 以逗号分开
            if i == 0:
                line[0] = 'a'  # 不知道为什么第一个数据乱码，用写字板打开是'a'

            line[0] = ord(line[0]) - 96
            line[1] = float(line[1]) - 48
            line[2] = ord(line[2]) - 96
            line[3] = float(line[3]) - 48
            line[4] = ord(line[4]) - 96
            line[5] = float(line[5]) - 48
            data[i, :] = line[:-1]

            if line[6][0] == 'd':
                label[i] = np.array([1, 0])
            else:
                label[i] = np.array([0, 1])
            i += 1
            if i == 28056:
                break

    ratioTraining = 0.4
    ratioValidation = 0.1
    ratioTesting = 0.5
    xTraining, xTesting, yTraining, yTesting = train_test_split(data, label, test_size=1 - ratioTraining,
                                                                random_state=0)  # 随机分配数据集
    xTesting, xValidation, yTesting, yValidation = train_test_split(xTesting, yTesting,
                                                                    test_size=ratioValidation / ratioTesting,
                                                                    random_state=0)
    # 拆分成测试集和验证集

    scaler = StandardScaler(copy=False)
    scaler.fit(xTraining)
    scaler.transform(xTraining)  # 标准归一化
    scaler.transform(xTesting)
    scaler.transform(xValidation)

    nn = NN(layer=[6, 20, 20, 20, 2], active_function='relu', learning_rate=0.01, batch_normalization=1,
            optimization_method='Adam',
            objective_function='Cross Entropy')

    option = Option()
    option.batch_size = 50
    option.iteration = 1

    iteration = 0
    maxAccuracy = 0
    totalAccuracy = []
    totalCost = []
    maxIteration = 20
    while iteration < maxIteration:
        iteration = iteration + 1
        nn = nn_train(nn, option, xTraining, yTraining)
        totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
        # plot(totalCost)
        (wrongs, accuracy) = nn_test(nn, xValidation, yValidation)
        totalAccuracy.append(accuracy)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            storedNN = nn

        cost = totalCost[iteration - 1]
        print(accuracy)
        print(totalCost[iteration - 1])

    subplot(2, 1, 1)
    plot(totalCost, color='red')
    title('Average Objective Function Value on the Training Set')

    subplot(2, 1, 2)
    plot(totalAccuracy, color='red')
    ylim([0.8, 1])
    title('Accuracy on the Validation Set')
    tight_layout(2)
    show()

    wrongs, accuracy = nn_test(storedNN, xTesting, yTesting)
    print('acc:', accuracy)

nn_testChess()
