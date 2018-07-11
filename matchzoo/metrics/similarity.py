# -*- coding: utf-8 -*-

from __future__ import division

from keras import backend as K

def dot(a, b):
    return K.batch_dot(a, b, axes=1)

def l2(a, b):
    return K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

def cosine(a, b):
    return dot(a, b) / K.maximum(K.sqrt(dot(a, a) * dot(b, b)), K.epsilon())

def polynomial(a, b, gamma=1, c=1, d=1):
    return (gamma * dot(a, b) + c) ** d

def sigmoid(a, b, gamma=1, c=1):
    return K.tanh(gamma * dot(a, b) + c)

def rbf(a, b, gamma=1):
    return K.exp(-1 * gamma * l2(a, b) ** 2)

def euclidean(a, b):
    return 1 / (1 + l2(a, b))

def exponential(a, b, gamma=1):
    return K.exp(-1 * gamma * l2(a, b))

def gesd(a, b, gamma=1, c=1):
    return euclidean(a, b) * sigmoid(a, b, gamma, c)

def aesd(a, b, gamma=1, c=1):
    return 0.5 * euclidean(a, b) + 0.5 * sigmoid(a, b, gamma, c)

