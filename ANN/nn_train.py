import numpy as np
from nn_forward import nn_forward
from nn_backpropagation import nn_backpropagation
from nn_applygradient import nn_applygradient

def nn_train(nn,option,train_x,train_y):
    iteration = option.iteration
    batch_size = option.batch_size
    m = train_x.shape[0]
    num_batches = m / batch_size
    for k in range(iteration): 
        kk = np.random.permutation(m)
        for l in range(int(num_batches)):
            batch_x = train_x[kk[l * batch_size : (l + 1) * batch_size], :] #(l+1)*batch_size也可以改成max((l+1)*batch_size, len(kk))
            batch_y = train_y[kk[l * batch_size : (l + 1) * batch_size], :]
            nn = nn_forward(nn,batch_x,batch_y)
            nn = nn_backpropagation(nn,batch_y)
            nn = nn_applygradient(nn)
            
    return nn