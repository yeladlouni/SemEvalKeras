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

class AvgNN(BasicModel):
    def __init__(self, config):
        super(AvgNN, self).__init__(config)
        self.__name = 'AvgNN'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 'embed_size' ]
        self.embed_trainable = config['train_embed']
        self.projection = config['projection']
        self.setup(config)
        if not self.check():
            raise TypeError('[AvgNN] parameter check wrong')
        print('[AvgNN] init done', end='\n')

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

        shared_dense = Dense(int(self.config['embed_size']), activation='linear')
        q_wproj = TimeDistributed(shared_dense)(q_embed)
        d_wproj = TimeDistributed(shared_dense)(d_embed)
        
        q_wproj = TimeDistributed(BatchNormalization())(q_wproj)
        d_wproj = TimeDistributed(BatchNormalization())(d_wproj)

        q_wproj = TimeDistributed(Activation('tanh'))(q_wproj)
        d_wproj = TimeDistributed(Activation('tanh'))(d_wproj)
        
        avg_layer = Lambda(function=lambda x: K.mean(x, axis=1))
        q_avg = avg_layer(q_wproj)
        d_avg = avg_layer(d_wproj)

        if self.config['projection']:
            q_avg, d_avg = Projection(features=self.config['embed_size'])([q_avg, d_avg])

        scores = MLPScorer(features=self.config['embed_size'])([q_avg, d_avg])
        model = Model(inputs=[query, doc], outputs=scores)
        return model
