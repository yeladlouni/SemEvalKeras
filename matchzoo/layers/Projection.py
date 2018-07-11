from keras import backend as K
from keras.engine.topology import Layer

from keras import layers
from keras import regularizers

import numpy as np

class Projection(Layer):
    def __init__(self, features, factor=0.2, n_layers=1, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=1e-4, dropout=0.5, **kwargs):
        self.features = features
        self.factor = factor
        self.n_layers = n_layers
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        super(Projection, self).__init__(**kwargs)        

    def build(self, input_shape):
        super(Projection, self).build(input_shape)

    def call(self, x):
        x1 = x[0]
        x2 = x[1]

        for i in range(self.n_layers):
            shared_dense = layers.Dense(int(self.features * self.factor), activation='linear',
                kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l2(self.kernel_regularizer))
            q_proj = layers.Activation(self.activation)(layers.BatchNormalization()(shared_dense(x1)))
            d_proj = layers.Activation(self.activation)(layers.BatchNormalization()(shared_dense(x2)))
            x1 = q_proj
            x2 = d_proj
            self.features = int(self.features * self.factor)

        dropout = self.dropout
        q_proj = layers.Dropout(dropout, noise_shape=(self.features,))(q_proj)
        d_proj = layers.Dropout(dropout, noise_shape=(self.features,))(d_proj)

        return [q_proj, d_proj]

    def compute_output_shape(self, input_shape):
        return [(None,self.features), (None,self.features)]

    
