# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax
from keras.regularizers import l2
import tensorflow as tf
from model import BasicModel
from utils.utility import *
from layers.Projection import Projection
from layers.MLPScorer import MLPScorer

class CNN(BasicModel):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.__name = 'CNN'
        
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 'embed_size' ]
        
        self.embed_trainable = config['train_embed']
        self.mode = concatenate
        self.activation = config['activation']
        self.init = config['init']
        self.reg = config['reg']
        self.dim = config['dim']
        self.dropout = config['dropout']
        self.max_len = 20
        self.features = config['embed_size']
        self.return_sequence = False
        self.projection = config['projection']

        self.setup(config)
        
        if not self.check():
            raise TypeError('[CNN] parameter check wrong')
        print('[CNN] init done', end='\n')

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
                
        q_cnn_res_list = []
        d_cnn_res_list = []
        tot_len = 0
        for fl, cd in enumerate(self.dim):
            nb_filter = int(self.features*cd)
            shared_conv = Convolution1D(input_shape=(None, self.max_len, self.features),
                        kernel_size=fl, filters=nb_filter, activation='linear',
                        kernel_regularizer=l2(self.reg), kernel_initializer=self.init)
            q_cnn_one = Activation(self.activation)(BatchNormalization()(shared_conv(q_embed)))
            d_cnn_one = Activation(self.activation)(BatchNormalization()(shared_conv(d_embed)))
            
            pool = MaxPooling1D(pool_size=int(self.max_len-fl+1))
            q_pool_one = pool(q_cnn_one)
            d_pool_one = pool(d_cnn_one)

            flatten = Flatten()

            q_out_one = flatten(q_pool_one)
            d_out_one = flatten(d_pool_one)

            q_cnn_res_list.append(q_out_one)
            d_cnn_res_list.append(d_out_one)
        
            tot_len += nb_filter

        q_cnn = Dropout(self.dropout, noise_shape=(tot_len,))(concatenate(q_cnn_res_list))
        d_cnn = Dropout(self.dropout, noise_shape=(tot_len,))(concatenate(d_cnn_res_list))
        
        if self.projection:
            q_cnn, d_cnn = Projection(self.features)([q_cnn, d_cnn])

        scores = MLPScorer(features=self.config['embed_size'])([q_cnn, d_cnn])
        model = Model(inputs=[query, doc], outputs=scores)
        return model
