# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import


import keras.backend as K
from keras import layers
from keras import regularizers
from keras import initializers
from keras import Model

from model import BasicModel
from layers import MLPScorer

class RankNet(BasicModel):
    def __init__(self, config):
        super(RankNet, self).__init__(config)
        self.__name = 'RankNet'
        
        self.query_max_len = config['text1_maxlen']
        self.doc_max_len = config['text1_maxlen']
        self.embed_trainable = False
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embed_size']
        self.embedding_matrix = config['embed']
        
        self.setup(config)
        
        if not self.check():
            raise TypeError('[RankNet] parameter check wrong')
        print('[RankNet] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):
        def getDivisor(x):
            return K.sqrt(K.sum(K.square(x),axis=-1,keepdims=True))

        def cosine(x): 
            a, b = x
            dividend = K.sum(a*b,axis=-1,keepdims=True)
            return dividend / (getDivisor(a) * getDivisor(b))

        query = layers.Input(name='query', shape=(self.query_max_len,))
        doc = layers.Input(name='doc', shape=(self.doc_max_len,))

        q_embed = layers.Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix], trainable=self.embed_trainable)(query)
        d_embed = layers.Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix], trainable=self.embed_trainable)(doc)
        
        q_pooled = layers.GlobalMaxPooling1D()(q_embed)
        d_pooled = layers.GlobalMaxPooling1D()(d_embed)

        match = layers.concatenate([q_pooled, d_pooled])

        output = layers.Dense(2, activation='softmax')(match)
        
        model = Model([query, doc], output)
        
        model.summary()
        return model


# convenience l2_norm function
def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm

def pairwise_cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A, K.permute_dimensions(B, (0,2,1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat =  num / den

    return dist_mat