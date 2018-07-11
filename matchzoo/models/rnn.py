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

class RNN(BasicModel):
    def __init__(self, config):
        super(RNN, self).__init__(config)
        self.__name = 'RNN'
        
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
            raise TypeError('[RNN] parameter check wrong')
        print('[RNN] init done', end='\n')

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
                
        if self.mode == concatenate:
            self.dim /= 2
        
        shared_rnn_f = self.rnn(int(self.features*self.dim), kernel_initializer=self.init, input_shape=(None, self.max_len, self.features), 
                        return_sequences=self.return_sequence)
        shared_rnn_b = self.rnn(int(self.features*self.dim), kernel_initializer=self.init, input_shape=(None, self.max_len, self.features),
                        return_sequences=self.return_sequence, go_backwards=True)
        
        q_rnn_f = shared_rnn_f(q_embed)
        d_rnn_f = shared_rnn_f(d_embed)
        
        q_rnn_b = shared_rnn_b(q_embed)
        d_rnn_b = shared_rnn_b(d_embed)
        
        q_rnn = Activation(self.activation)(BatchNormalization()(self.mode([q_rnn_f, q_rnn_b])))
        d_rnn = Activation(self.activation)(BatchNormalization()(self.mode([d_rnn_f, d_rnn_b])))
        
        if self.mode == concatenate:
            self.dim *= 2
            
        q_rnn = Dropout(self.dropout, noise_shape=(int(self.features*self.dim),))(q_rnn)
        d_rnn = Dropout(self.dropout, noise_shape=(int(self.features*self.dim),))(d_rnn)
        
        if self.projection:
            q_rnn, d_rnn = Projection(self.features)([q_rnn, d_rnn])

        scores = MLPScorer(features=self.config['embed_size'])([q_rnn, d_rnn])
        model = Model(inputs=[query, doc], outputs=scores)
        return model
