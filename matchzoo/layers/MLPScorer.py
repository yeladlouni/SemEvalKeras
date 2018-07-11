from keras import backend as K
from keras.engine.topology import Layer

from keras import layers
from keras import regularizers


import numpy as np

class MLPScorer(Layer):
    def __init__(self, features, n_layers=2, activation='sigmoid', kernel_regularizer='1e-4', **kwargs):
        self.features = features
        self.n_layers = n_layers
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        super(MLPScorer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MLPScorer, self).build(input_shape)

    def call(self, x):
        sum_vec = layers.add(x)
        mul_vec = layers.multiply(x)

        mlp_input = layers.concatenate([sum_vec, mul_vec])

        # Ddim may be either 0 (no hidden layer), scalar (single hidden layer) or
        # list (multiple hidden layers)
        if self.n_layers == 0:
            n_layers = []
        elif not isinstance(self.n_layers, list):
            n_layers = [self.n_layers]

        if n_layers:
            for i, n_layer in enumerate(n_layers):
                shared_dense = layers.Dense(int(self.features*n_layer), kernel_regularizer=regularizers.l2(self.kernel_regularizer), activation='linear')
                mlp_input = layers.Activation('tanh')(shared_dense(mlp_input))

        shared_dense = layers.Dense(1, kernel_regularizer=regularizers.l2(self.kernel_regularizer), activation=self.activation)

        mlp_out = shared_dense(mlp_input)

        return mlp_out

    def compute_output_shape(self, input_shape):
        return (None,1)

    