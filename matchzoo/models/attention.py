# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

import keras.backend as K
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax
from model import BasicModel
from utils.utility import *

from layers.Projection import Projection
from layers.MLPScorer import MLPScorer

class Attention(BasicModel):
    def __init__(self, config):
        super(Attention, self).__init__(config)
        self.__name = 'Attention'
        
        self.check_list = [ 'text1_maxlen', 'text2_maxlen', 'embed_size' ]
        
        self.embed_trainable = config['train_embed']
        self.features = config['embed_size']
        self.mode = concatenate
        self.activation = config['activation']
        self.init = config['init']
        self.rnn = GRU
        self.dim = config['dim']
        self.dropout = config['dropout']
        self.max_len = 20
        self.return_sequence = False
        self.projection = config['projection']
        self.reg = 1e-4
        self.padding = 40

        self.setup(config)
        
        if not self.check():
            raise TypeError('[Attention] parameter check wrong')
        print('[Attention] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.config.update(config)
    
    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'])
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        q_rnn, d_rnn, nc_rnn = rnn_input(self.features, sdim=5, dropout=self.dropout, rnnbidi_mode=concatenate, rnn=LSTM, 
                            rnnact='tanh', rnninit='glorot_uniform', inputs=[q_embed, d_embed], 
                            return_sequence=True, padding=self.padding)
        # calculate the sentence vector on X side using Convolutional Neural Networks
        q_aggreg, nc_cnn = aggregate(self.features, self.padding, l2reg=self.reg, cnninit='glorot_uniform', 
                                    cnnact='relu', input_dim=nc_rnn, inputs=q_rnn, 
                                    cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}, padding=self.padding)
        
        # re-embed X,Y in attention space
        adim = int(1)
        shared_dense_q = Dense(adim, kernel_regularizer=l2(self.reg), kernel_initializer='glorot_uniform')
        q_aggreg_attn = BatchNormalization()(shared_dense_q(q_aggreg))


        shared_dense_s = Dense(adim, kernel_regularizer=l2(self.reg), kernel_initializer='glorot_uniform')
        d_attn = TimeDistributed(shared_dense_s)(d_rnn)
        d_attn = TimeDistributed(BatchNormalization())(d_attn)
        # apply an attention function on Y side by producing an vector of scalars denoting the attention for each token
        d_foc = focus(self.features, q_aggreg_attn, d_attn, d_rnn, 5, adim, self.reg, padding=self.padding)

        d_aggreg, nc_cnn = aggregate(self.features, self.padding, l2reg=self.reg, cnninit='glorot_uniform', 
                                    cnnact='relu', input_dim=nc_rnn, inputs=d_foc, 
                                    cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2}, padding=self.padding)
        
        if self.projection:
            q_val, d_val = Projection(nc_cnn)([q_aggreg, d_aggreg])

        scores = MLPScorer(self.features)([q_val, d_val])
        #scores = concatenate([q_val, d_val])
        #scores = Dense(1, activation='sigmoid')(scores)

        model = Model(inputs=[query, doc], outputs=scores)
        return model

def aggregate(pad, dropout=1/2, l2reg=1e-4, cnninit='glorot_uniform', cnnact='relu',
        cdim={1: 1/2, 2: 1/2, 3: 1/2, 4: 1/2, 5: 1/2, 6: 1/2, 7: 1/2}, inputs=None, input_dim=304, padding=10):
    cnn_res_list = []
    tot_len = 0
    for fl, cd in cdim.items():
        nb_filter = int(input_dim*cd)
        shared_conv = Convolution1D(input_shape=(None, padding, input_dim),
                    kernel_size=fl, filters=nb_filter, activation='linear',
                    kernel_regularizer=l2(l2reg), kernel_initializer=cnninit)
        cnn_res = Activation(cnnact)(BatchNormalization()(shared_conv(inputs)))

        pool = MaxPooling1D(pool_size=int(padding-fl+1))
        cnn_res = pool(cnn_res)
        flatten = Flatten()
        cnn_res = flatten(cnn_res)

        cnn_res_list.append(cnn_res)

        tot_len += nb_filter

    aggreg = Dropout(dropout, noise_shape=(tot_len,))(concatenate(cnn_res_list))

    return (aggreg, tot_len)

def focus(N, input_aggreg, input_seq, orig_seq, sdim, awidth, l2reg, padding=10):
    repeat_vec = RepeatVector(padding)
    input_aggreg_rep = repeat_vec(input_aggreg)
    attn = Activation('tanh')(add([input_aggreg_rep, input_seq]))

    shared_dense = Dense(1, kernel_regularizer=l2(l2reg))
    attn = TimeDistributed(shared_dense)(attn)

    flatten = Flatten()
    attn = flatten(attn)

    attn = Activation('softmax')(attn)
    attn = RepeatVector(int(N*sdim))(attn)
    attn = Permute((2,1))(attn)
    output = multiply([orig_seq, attn])

    return output

def rnn_input(N, dropout=3/4, sdim=2, rnn=GRU, rnnact='tanh', rnninit='glorot_uniform', rnnbidi_mode=add, 
              inputs=None, return_sequence=True, padding=10):
    if rnnbidi_mode == concatenate:
        sdim /= 2
    shared_rnn_f = rnn(int(N*sdim), kernel_initializer=rnninit, input_shape=(None, padding, N), 
                       return_sequences=return_sequence)
    shared_rnn_b = rnn(int(N*sdim), kernel_initializer=rnninit, input_shape=(None, padding, N),
                       return_sequences=return_sequence, go_backwards=True)
    qi_rnn_f = shared_rnn_f(inputs[0])
    si_rnn_f = shared_rnn_f(inputs[1])
    qi_rnn_b = shared_rnn_b(inputs[0])
    si_rnn_b = shared_rnn_b(inputs[1])
    
    qi_rnn = Activation(rnnact)(BatchNormalization()(rnnbidi_mode([qi_rnn_f, qi_rnn_b])))
    si_rnn = Activation(rnnact)(BatchNormalization()(rnnbidi_mode([si_rnn_f, si_rnn_b])))
    
    if rnnbidi_mode == concatenate:
        sdim *= 2
        
    qi_rnn = Dropout(dropout, noise_shape=(int(N*sdim),))(qi_rnn)
    si_rnn = Dropout(dropout, noise_shape=(int(N*sdim),))(si_rnn)
    
    return (qi_rnn, si_rnn, int(N*sdim))