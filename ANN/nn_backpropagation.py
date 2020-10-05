import numpy as np

def nn_backpropagation(nn, batch_y) :
    batch_y = batch_y.T
    m = nn.a[0].shape[1]
    nn.theta[1] = 0
    f = nn.output_function
    if f == 'sigmoid' :
        nn.theta[nn.depth-1] = -(batch_y - nn.a[nn.depth-1]) * nn.a[nn.depth-1] * (1 - nn.a[nn.depth-1])
    if f == 'tanh' :
        nn.theta[nn.depth-1] = -(batch_y - nn.a[nn.depth-1]) * (1 - nn.a[nn.depth-1]**2)
    if f == 'softmax' :
        y = np.dot(nn.W[nn.depth - 2], nn.a[nn.depth - 2]) + np.tile(nn.b[nn.depth - 2], (1, m))
        nn.theta[nn.depth-1] = nn.a[nn.depth-1] - batch_y

    if nn.batch_normalization :
        x = np.dot(nn.W[nn.depth - 2], nn.a[nn.depth -2]) + np.tile(nn.b[nn.depth - 2], (1, m))
        x = (x - np.tile(nn.E[nn.depth -2], (1, m))) / np.tile(nn.S[nn.depth -2] + 0.0001*np.ones(nn.S[nn.depth - 2].shape), (1, m))
        temp = nn.theta[nn.depth-1] * x
        nn.Gamma_grad[nn.depth - 2] = sum(np.mean(temp, axis = 1))
        nn.Beta_grad[nn.depth - 2] = sum(np.mean(nn.theta[nn.depth-1], axis = 1))
        nn.theta[nn.depth - 1] = nn.Gamma[nn.depth - 2]*nn.theta[nn.depth-1] / np.tile((nn.S[nn.depth - 2] + 0.0001), (1, m))

    nn.W_grad[nn.depth - 2] = np.dot(nn.theta[nn.depth-1], nn.a[nn.depth - 2].T) / m + nn.weight_decay*nn.W[nn.depth - 2]
    nn.b_grad[nn.depth - 2] = np.array([np.sum(nn.theta[nn.depth-1], axis=1) / m]).T 
    #因为np.sum()返回维度为(n,)，会让之后的加法操作错误，所以要转换为(n,1)维度矩阵，下面的也是一样
    
    f = nn.active_function
    if f == 'sigmoid':
        if nn.encoder == 0 :
            for ll in range(1, nn.depth - 1) :
                k = nn.depth - ll-1
                nn.theta[k] = np.dot(nn.W[k].T, nn.theta[k + 1])*nn.a[k]* (1 - nn.a[k])
                if nn.batch_normalization :
                    x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
                    x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001*np.ones(nn.S[k - 1].shape), (1, m))
                    temp = nn.theta[k]*x
                    nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis = 1))
                    nn.Beta_grad[k - 1] = sum(np.mean(nn.theta[k], axis = 1))
                    nn.theta[k] = (nn.Gamma[k - 1]* nn.theta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
                    pass

                nn.W_grad[k - 1] = np.dot(nn.theta[k], nn.a[k - 1].T) / m + nn.weight_decay*nn.W[k - 1]
                nn.b_grad[k - 1] = np.array([np.sum(nn.theta[k], axis = 1) / m]).T

        else:
            #encoder完全按照matlab的NN，但貌似是有错误的，用encoder会报错，因为theta[2]（对应matlab的theta{3}）没有赋值
            roj = np.array([np.sum(nn.a[1], axis = 1) / m]).T 
            temp = (-nn.sparsity / roj + (1 - nn.sparsity) / (1 - roj))
            nn.theta[1] = (np.dot(nn.W[1].T, nn.theta[2]) + nn.beta*repmat(temp, 1, m))*M
            nn.W_grad[0] = np.dot(nn.theta[1], nn.a[0].T) / m + nn.weight_decay*nn.W[0]
            nn.b_grad[0] = np.array([np.sum(nn.theta[1], axis = 1) / m]).T
            

    elif f == 'tanh':
        for ll in range(1, nn.depth-1) :
            if nn.encoder == 0 :
                k = nn.depth - ll-1 
                nn.theta[k] = np.dot(nn.W[k].T,nn.theta[k + 1])*(1 - nn.a[k]**2)
                if nn.batch_normalization :
                    x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
                    x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001*np.ones(nn.S[k - 1].shape), (1, m))
                    temp = nn.theta[k]*x
                    nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis = 1))
                    nn.Beta_grad[k - 1] = sum(np.mean(nn.theta[k], axis = 1))
                    nn.theta[k] = (nn.Gamma[k - 1]* nn.theta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
                    pass

                nn.W_grad[k - 1] = np.dot(nn.theta[k], nn.a[k - 1].T) / m + nn.weight_decay*nn.W[k - 1]
                nn.b_grad[k - 1] = np.array([np.sum(nn.theta[k], axis = 1) / m]).T

            else:
                roj = np.array([np.sum(nn.a[1], axis = 1) / m]).T
                temp = (-nn.sparsity / roj + (1 - nn.sparsity) / (1 - roj))
                nn.theta[1] = (np.dot(nn.W[1].T, nn.theta[2]) + nn.beta*repmat(temp, 1, m))*M
                nn.W_grad[0] = np.dot(nn.theta[1], nn.a[0].T) / m + nn.weight_decay*nn.W[0]
                nn.b_grad[0] = np.array([np.sum(nn.theta[1], axis = 1) / m]).T

    elif f == 'relu':
        if nn.encoder == 0 :
            for ll in range(1, nn.depth - 1) :
                k = nn.depth - ll-1
                nn.theta[k] = np.dot(nn.W[k].T,nn.theta[k + 1])* (nn.a[k] > 0)
                if nn.batch_normalization :
                    x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
                    x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001*np.ones(nn.S[k - 1].shape), (1, m))
                    temp = nn.theta[k]*x
                    nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis = 1))
                    nn.Beta_grad[k - 1] = sum(np.mean(nn.theta[k], axis = 1))
                    nn.theta[k] = (nn.Gamma[k - 1]* nn.theta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
                    pass

                nn.W_grad[k - 1] = np.dot(nn.theta[k], nn.a[k - 1].T) / m + nn.weight_decay*nn.W[k - 1]
                nn.b_grad[k - 1] = np.array([np.sum(nn.theta[k], axis = 1) / m]).T

        else:
            roj = np.array([np.sum(nn.a[1], axis = 1) / m]).T
            temp = (-nn.sparsity / roj + (1 - nn.sparsity) / (1 - roj))
            M = np.maximum(nn.a[1], 0)
            M = M / np.maximum(M, 0.001)

            nn.theta[1] = (np.dot(nn.W[1].T, nn.theta[2]) + nn.beta*repmat(temp, 1, m))*M
            nn.W_grad[0] = np.dot(nn.theta[1], nn.a[0].T) / m + nn.weight_decay*nn.W[0]
            nn.b_grad[0] = np.array([np.sum(nn.theta[1], axis = 1) / m]).T
    return nn