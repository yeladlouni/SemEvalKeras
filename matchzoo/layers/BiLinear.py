from keras import backend as K
from keras.engine.topology import Layer

from keras import layers
from keras import regularizers

import numpy as np

class BiLinear(Layer): 
    def __init__(self, adim, qlen, alen, dropout, pfx, **kwargs): 
        self.adim = adim 
        self.qlen = qlen
        self.alen = alen
        self.dropout = dropout
        self.pfx = pfx
        super(BiLinearLayer, self).__init__(**kwargs) 
 
    def build(self, input_shape): 
        mean = 0.0 
        std = 1.0 
        # U : adim*adim 
        adim = self.adim 
        initial_U_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(adim,adim))
        self.U = K.variable(initial_U_values, name='bilinear'+self.pfx)
        self.trainable_weights = [self.U] 
        
    def call(self, inputs, mask=None): 
        if type(inputs) is not list or len(inputs) <= 1: 
            raise Exception('BiLinearLayer must be called on a list of tensors ' 
                            '(at least 2). Got: ' + str(inputs)) 
        Q = inputs[0]
        A = inputs[1]
        QU = K.dot(Q,self.U) # shape=(None, pad, adim)
        AT = Permute((2,1))(A) # shape=(None, adim, pad)
        QUA_T = K.batch_dot(QU, AT)
        QUA_T = K.tanh(QUA_T) # shape=(pad, pad)
        return QUA_T

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.qlen, self.alen)