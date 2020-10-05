import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """返回对应的概率值"""
    exp_x = np.exp(x)
    softmax_x = np.zeros(x.shape,dtype=float)
    for i in range(len(x[0])):
        softmax_x[:,i] = exp_x[:,i] / (exp_x[0,i] + exp_x[1,i])
        
    return softmax_x 