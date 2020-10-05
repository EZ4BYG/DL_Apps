import numpy as np
from function import sigmoid, softmax

def nn_predict(nn, batch_x):
    batch_x = batch_x.T 
    m = batch_x.shape[1]
    nn.a[0] = batch_x 
    for k in range(1, nn.depth):
        y = np.dot(nn.W[k-1], nn.a[k-1]) + np.tile(nn.b[k-1], (1, m))
        if nn.batch_normalization:
            y = (y - np.tile(nn.E[k-1], (1, m))) / np.tile(nn.S[k-1]+0.0001*np.ones(nn.S[k-1].shape), (1, m)) 
            y = nn.Gamma[k-1]*y + nn.Beta[k-1] 

        if k == nn.depth-1:
            f = nn.output_function
            if f == 'sigmoid':
                nn.a[k] = sigmoid(y) 
            elif f == 'tanh':
                nn.a[k] = np.tanh(y) 
            elif f == 'relu':
                nn.a[k] = np.maximum(y, 0) 
            elif f == 'softmax':
                nn.a[k] = softmax(y) 

        else:
            f = nn.active_function
            if f == 'sigmoid':
                nn.a[k] = sigmoid(y) 
            elif f == 'tanh':
                nn.a[k] = np.tanh(y) 
            elif f == 'relu':
                nn.a[k] = np.maximum(y, 0) 
            
    return nn