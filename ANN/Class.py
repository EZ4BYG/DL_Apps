import numpy as np

class Option:
    def __init__(self):
        self.batch_size = 0
        self.iteration = 0
        
class NN:
    def __init__(self, **arg):
        init = {'layer':[],
                'active_function':'sigmoid', 
                'output_function':'sigmoid', 
                'learning_rate':1.5, 
                'weight_decay':0,
                'cost':{}, 
                'encoder':0,
                'sparsity':0.03,
                'beta':3,
                'batch_normalization':0,
                'grad_squared':0,
                'r':0,
                'optimization_method':'normal',
                'objective_function':'MSE'
               }
        
        param = dict() #字典结构实现参数列表
        param.update(init)
        param.update(arg)
        
        self.size = param['layer'] #取出字典的值初始化网络参数
        self.depth = len(self.size)
        self.active_function = param['active_function']
        self.output_function = param['output_function']
        self.learning_rate = param['learning_rate']
        self.weight_decay = param['weight_decay']
        self.encoder = param['encoder']
        self.sparsity = param['sparsity']
        self.beta = param['beta']
        self.cost = param['cost']
        self.batch_normalization = param['batch_normalization']
        self.grad_squared = param['grad_squared']
        self.r = param['r']
        self.optimization_method = param['optimization_method']
        self.objective_function = param['objective_function']
        self.a = dict()

        if self.objective_function == 'Cross Entropy':
            self.output_function = 'softmax'

        self.W = dict(); self.b = dict(); self.vW = dict(); self.vb = dict() #python必须要先初始化字典才能用
        self.rW = dict(); self.rb = dict(); self.sW = dict(); self.sb = dict() #注意要单独初始化，否则它们以后也一直是一样的
        self.E = dict(); self.S = dict(); self.Gamma = dict(); self.Beta = dict()
        self.vGamma = dict(); self.rGamma = dict(); self.vBeta = dict(); self.rBeta = dict(); 
        self.sGamma = dict(); self.sBeta = dict(); self.W_grad = dict(); self.b_grad = dict(); self.theta = dict()
        self.Gamma_grad = dict(); self.Beta_grad = dict()
        
        for k in range(self.depth - 1):
            width = self.size[k]
            height = self.size[k + 1]
            #self.W{ k } = (np.random.rand(height, width) - 0.5) * 2 * np.sqrt(6 / (height + width + 1)) - np.sqrt(6 / (height + width + 1));

            self.W[k] = 2 * np.random.rand(height, width) / np.sqrt(width) - 1 / np.sqrt(width)
            
            #self.W{ k } = 2 * np.random.rand(height, width) - 1;
            #Xavier initialization
            if self.active_function == 'relu':
                self.b[k] = np.random.rand(height, 1) + 0.01
            else:
                self.b[k] = 2 * np.random.rand(height, 1) / np.sqrt(width) - 1 / np.sqrt(width)

            #parameters for moments
            method = self.optimization_method

            if method == 'Momentum':
                self.vW[k] = np.zeros((height, width), dtype=float)
                self.vb[k] = np.zeros((height, 1), dtype=float)

            if method == 'AdaGrad' or method == 'RMSProp' or method == 'Adam':
                self.rW[k] = np.zeros((height, width), dtype=float)
                self.rb[k] = np.zeros((height, 1), dtype=float)

            if method == 'Adam':
                self.sW[k] = np.zeros((height, width), dtype=float)
                self.sb[k] = np.zeros((height, 1), dtype=float)

            #parameters for batch normalization.
            if self.batch_normalization:
                self.E[k] = np.zeros((height, 1), dtype=float)
                self.S[k] = np.zeros((height, 1), dtype=float)
                self.Gamma[k] = 1
                self.Beta[k] = 0

                if  method == 'Momentum':
                    self.vGamma[k] = 1
                    self.vBeta[k] = 0

                if method == 'AdaGrad' or method == 'RMSProp' or method == 'Adam':
                    self.rW[k] = np.zeros((height, width), dtype=float)
                    self.rb[k] = np.zeros((height, 1), dtype=float)
                    self.rGamma[k] = 0
                    self.rBeta[k] = 0

                if  method == 'Adam':
                    self.sGamma[k] = 1
                    self.sBeta[k] = 0

                self.vecNum = 0
                
            self.W_grad[k] = np.zeros((height, width), dtype=float)