import numpy as np
from function import softmax, sigmoid

def nn_forward(nn, batch_x, batch_y):
    s = len(nn.cost) + 1 
    batch_x = batch_x.T 
    batch_y = batch_y.T 
    m = batch_x.shape[1]
    nn.a[0] = batch_x 

    cost2 = 0 
    for k in range(1, nn.depth):
        y = np.dot(nn.W[k-1], nn.a[k-1]) + np.tile(nn.b[k-1], (1, m)) #np.tile就是matlab中的repmat(replicate matrix)
        
        if nn.batch_normalization:
            nn.E[k-1] = nn.E[k-1]*nn.vecNum + np.array([np.sum(y, axis=1)]).T
            nn.S[k-1] = nn.S[k-1]**2 * (nn.vecNum - 1) + np.array([(m - 1)*np.std(y,ddof=1,axis=1)** 2]).T #ddof=1计算无偏估计
            nn.vecNum = nn.vecNum + m 
            nn.E[k-1] = nn.E[k-1] / nn.vecNum 
            nn.S[k-1] = np.sqrt(nn.S[k-1] / (nn.vecNum - 1)) 
            y = (y - np.tile(nn.E[k-1], (1, m))) / np.tile(nn.S[k-1]+0.0001*np.ones(nn.S[k-1].shape), (1, m)) 
            y = nn.Gamma[k-1]*y + nn.Beta[k-1] 

        if k == nn.depth - 1:
            f = nn.output_function
            if f == 'sigmoid' :
                nn.a[k] = sigmoid(y)
            elif f == 'tanh' :
                nn.a[k] = np.tanh(y)
            elif f == 'relu' :
                nn.a[k] = np.maximum(y, 0)
            elif f == 'softmax' :
                nn.a[k] = softmax(y)

        else:
            f = nn.active_function
            if f == 'sigmoid' :
                nn.a[k] = sigmoid(y)
            elif f == 'tanh' :
                nn.a[k] = np.tanh(y)
            elif f == 'relu' :
                nn.a[k] = np.maximum(y, 0)

        cost2 = cost2 + np.sum(nn.W[k-1]**2)

    if nn.encoder == 1:
        roj = np.sum(nn.a[2], axis=1) / m 
        nn.cost[s] = 0.5 * np.sum((nn.a[k] -batch_y)**2) / m + 0.5 * nn.weight_decay * cost2 + 3 * sum(nn.sparsity * np.log(nn.sparsity / roj) + (1-nn.sparsity) * np.log((1-nn.sparsity) / (1-roj)))
    else:
        if nn.objective_function == 'MSE':
            nn.cost[s] = 0.5 / m * sum(sum((nn.a[k] -batch_y)** 2)) + 0.5 * nn.weight_decay * cost2 
        elif nn.objective_function == 'Cross Entropy':
            nn.cost[s] = -0.5 * sum(sum(batch_y*np.log(nn.a[k]))) / m + 0.5 * nn.weight_decay * cost2 
    # nn.cost[s]
    
    return nn