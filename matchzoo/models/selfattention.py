# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax
import tensorflow as tf
from model import BasicModel
from utils.utility import *
from layers.Projection import Projection
from layers.MLPScorer import MLPScorer

class SelfAttention(BasicModel):
    def __init__(self, config):
        super(SelfAttention, self).__init__(config)
        self.__name = 'SelfAttention'
        
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 'embed_size' ]
        
        self.embed_trainable = config['train_embed']
        self.mode = concatenate
        self.activation = config['activation']
        self.init = config['init']
        self.rnn = GRU
        self.dim = config['dim']
        self.dropout = config['dropout']
        self.max_len = 20
        self.features = config['embed_size']
        self.return_sequence = False
        self.projection = config['projection']

        self.setup(config)
        
        if not self.check():
            raise TypeError('[SelfAttention] parameter check wrong')
        print('[SelfAttention] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
                
        shared_dense = Dense(64, activation='linear') 
        q_key = TimeDistributed(shared_dense)(q_embed)
        d_key = TimeDistributed(shared_dense)(d_embed) 

        q_key = TimeDistributed(BatchNormalization())(q_key)
        d_key = TimeDistributed(BatchNormalization())(d_key)

        q_key = TimeDistributed(Activation('relu'))(q_key)
        q_key = TimeDistributed(Activation('relu'))(d_key)
        
        # one-more 1x1 spartial convolution
        shared_dense_attn = Dense(1, activation='linear') 
        q_matching = TimeDistributed(shared_dense_attn)(q_key)
        d_matching = TimeDistributed(shared_dense_attn)(d_key)
        
        # get attn values
        flatten = Flatten()
        q_matching = Activation('softmax')(flatten(q_matching))
        q_matching = RepeatVector(self.features)(q_matching)
        q_matching = Permute((2,1))(q_matching)
        d_matching = Activation('softmax')(flatten(d_matching))
        d_matching = RepeatVector(self.features)(d_matching)
        d_matching = Permute((2,1))(d_matching)

        # q, sentence updates
        q_val = multiply([q_matching, q_embed])
        d_val = multiply([d_matching, d_embed])

        # weighted_averaging
        avg_layer = Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape:(shape[0],) + shape[2:])
        q_val = avg_layer(q_val)
        d_val = avg_layer(d_val)

        if self.projection:
            q_val, d_val = Projection(self.features)([q_val, d_val])

        #scores = MLPScorer(self.features)([q_val, d_val])
        scores = concatenate([q_val, d_val])
        scores = Dense(1, activation='sigmoid')(scores)

        model = Model(inputs=[query, doc], outputs=scores)
        return model
