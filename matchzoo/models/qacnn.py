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
from metrics.similarity import *
from layers.SparseFullyConnectedLayer import *

class QACNN(BasicModel):
    def __init__(self, config):
        super(QACNN, self).__init__(config)
        self.__name = 'QACNN'
        self.check_list = [ 'vocab_size', 'hidden_sizes', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[QACNN] parameter check wrong')
        print('[QACNN] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_sizes', [300, 128])
        self.set_default('dropout_rate', 0.5)
        self.set_default('reg_rate', 0.0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=True)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        # cnn = Conv1D(512, 1)
        # q_cnn = cnn(q_embed)
        # show_layer_info('CNN', q_cnn)
        # d_cnn = cnn(d_embed)
        # show_layer_info('CNN', d_cnn)

        # pooling = GlobalMaxPool1D()
        # q_pool = pooling(q_cnn)
        # show_layer_info('Pooling', q_pool)
        # d_pool = pooling(d_cnn)
        # show_layer_info('Pooling', d_pool)

        bigru = Bidirectional(LSTM(32))
        q_encoded = bigru(q_embed)
        show_layer_info('BiGRU', q_encoded)
        d_encoded = bigru(d_embed)
        show_layer_info('BiGRU', d_encoded)
       
        sim = Dot(axes=[1,1], normalize=True)([q_encoded, d_encoded])
        #sim = Lambda(lambda x: aesd(x[0], x[1]))([q_pool, d_pool])        

        model = Model(inputs=[query, doc], outputs=[sim])
        return model
