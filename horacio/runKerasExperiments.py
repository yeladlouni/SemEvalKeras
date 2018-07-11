# -*- coding: utf-8 -*-
# !/pkg/ldc/bin/python2.7
#-----------------------------------------------------------------------------
# Name:        runKerasExperiments.py
#
# Author:      Horacio
#
# Created:     2018/01/18
# Run Keras experiments 
#-----------------------------------------------------------------------------

import keras
print keras.__version__
from keras import models
from keras import layers
from keras.models import Input, Model
from keras.layers import Dense, Dropout

import string
import re
from types import IntType,ListType,StringType,UnicodeType,BooleanType,TupleType
from os import listdir, fsync
import itertools
import numpy as np
from scipy import stats
from sklearn import svm, linear_model, cross_validation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import pickle
import gzip
import os
import pandas as pd
from pandas import DataFrame, ExcelWriter
import time


user = 'horacioLinux'
##user = 'horacioLocal'
if user == 'horacioWindowsLSI':
    semevalDir = 'L:/NQ/intercambio/pfc/ifrane/CQA/data/semeval2017/'
    semevalDir = 'C:/Users/horacio/Desktop/skater/CQA/data/semeval2017/semeval2017T3/Arabic/'
    evaluationsDir = 'L:/NQ/intercambio/pfc/ifrane/CQA/doc/SemEval2017_task3_submissions_and_scores/'
    gold = evaluationsDir+'_gold/SemEval2017-Task3-CQA-MD-test.xml.subtaskD.relevancy'
elif user == 'horacioLocal':
    semevalDir = 'C:/Users/horacio/Desktop/skater/CQA/data/semeval2017/'
    semevalDir2 = 'C:/Users/horacio/Desktop/skater/CQA/data/semeval2017/semeval2017T3/Arabic/'
    evaluationsDir = 'C:/Users/horacio/Desktop/skater/CQA/SemEval2017_task3_submissions_and_scores/'
    gold = evaluationsDir+'_gold/SemEval2017-Task3-CQA-MD-test.xml.subtaskD.relevancy'
    cqaDir = 'C:/Users/horacio/Desktop/skater/CQA/data/'
    cqaDirDoc = 'C:/Users/horacio/Desktop/skater/CQA/doc/'
elif user == 'horacioLinux':
    cqaDir = '/home/horacio/Public/CQA/data/'
    cqaDirDoc = '/home/horacio/Public/CQA/docs/'
    cqaYassineDir = '/home/horacio/Public/CQA_Yassine/data/'
    materialDir = '/home/horacio/Public/material/data/'
    theanoDir = '/home/horacio/Public/material/testsTheano/data/'
    evaluationsDir = '/home/horacio/Public/CQA/data/semeval/semeval2017/SemEval2017_task3_submissions_and_scores/'
    gold = evaluationsDir+'_gold/SemEval2017-Task3-CQA-MD-test.xml.subtaskD.relevancy'
    semevalDir = '/home/horacio/Public/CQA/data/semeval/semeval2017/semeval2017T3/Arabic/'

##globals

VALIDATION_SPLIT = 0.2
batch_size = 32
epochs = 5
clS = ['KerasLogReg','KerasLinearReg','KerasSimpleFNN','KerasDeep2FNN', 'KerasDeep3FNN', 'KerasDeep4FNN']


##classes

##functions

def gunzipFile(inF):
    f_out = open("/tmp/aaa.pic", 'wb')
    f_in = gzip.open(inF, 'rb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    return "/tmp/aaa.pic"

def loadDataSet(inF):
    print 'loading parameters'
    f = gunzipFile(cqaDir+inF+"_parameters.pic.gz")
    inF1 = open(f,'rb')
    parameters = pickle.load(inF1)
    inF1.close()
    print 'loading y'
    f = gunzipFile(cqaDir+inF+"_y.pic.gz")
    inF1 = open(f,'rb')
    y = pickle.load(inF1)
    inF1.close()
    print 'loading idS'
    f = gunzipFile(cqaDir+inF+"_idS.pic.gz")
    inF1 = open(f,'rb')
    idS = pickle.load(inF1)
    inF1.close()
    print 'loading X'
    fS = listdir(cqaDir)
    fS = filter(lambda x:x.startswith(inF),fS)
    fS = filter(lambda x:re.search("_X_.*\.pic.gz",x),fS)
    fS.sort(key = lambda x:int(re.search("_X_(.*)\.pic",x).group(1)))
    f = fS[0]; print 'loading', f;
    f1 = gunzipFile(cqaDir+f)
    inF1 = open(f1,'rb'); X = pickle.load(inF1); inF1.close()
    for f in fS[1:]:
        print 'loading', f;
        f1 = gunzipFile(cqaDir+f)
        inF1 = open(f1); Z = pickle.load(inF1); inF1.close()
        X = np.row_stack([X,Z])
        print len(Z), len(X)
    return parameters, X, y, idS

def iniProcess():
    global mode, semQs, lang
    print 'available datasets'
    fS = filter(lambda x: re.match('^.+\_[0-9]+\.pic.gz$',x),listdir(cqaDir))
    fS = list(set(map(lambda x: re.match('^(.+)\_X\_[0-9]+\.pic.gz$',x).group(1), fS)))
    fS.sort()
    for i in fS:
        print i
    return fS
        
                        
def preProcessId(id):
    global parameters, X, y, idS, train_set, test_set, x_train, y_train, x_test, y_test, num_classes
    parameters, X, y, idS = loadDataSet(id)
    parameters['numFeatures']=X.shape[1]
    for i in parameters:
        print i, parameters[i]
    print 'X', X.shape
    print 'y', y.shape
    print 'idS', len(idS)
    train_set, test_set = splitDataset()
    (x_train, y_train) = train_set
    (x_test, y_test) = test_set
    num_classes =  1
    print num_classes, 'classes'
##    y_train = keras.utils.to_categorical(y_train, num_classes)
##    y_test = keras.utils.to_categorical(y_test, num_classes)

def splitDataset():
    global X, y, idS, parameters, VALIDATION_SPLIT
    tS = int(VALIDATION_SPLIT * parameters['numSamples'])
    lS = parameters['numSamples'] - tS
    print ('splitting dataset into', lS, 'for training', tS, 'for test')
    train_set = (X[:lS],y[:lS])
    test_set = (X[lS:],y[lS:])
    print 'train', train_set[0].shape, train_set[1].shape
    print 'test', test_set[0].shape, test_set[1].shape
    return train_set, test_set

def evaluateClassifier(cl, y_pred, y_test):
    try:
        p = precision_score(y_pred, y_test)
    except:
        p=0
    try:
        r = recall_score(y_pred, y_test)
    except:
        r=0
    try:
        f1 = f1_score(y_pred, y_test)
    except:
        f1=0
    try:
        a = accuracy_score(y_pred, y_test)
    except:
        a=0
    try:
        m = average_precision_score(y_pred, y_test)
    except:
        m=0
    print 'precision', p
    print 'recall', r
    print 'f1', f1
    print 'accuracy', a
    print 'map', m
    return p, r, f1, a, m

def globalEvaluation(whichClassifiers, epochs, version, modes):
    global fS, allResults, log, results
    log = open(cqaDir+'log.txt','w')
    startTime= int(time.time())
    allResults = {}
    c=0
    for cl in whichClassifiers:
        print 'applying', cl, 'classifier'
        results = DataFrame(columns=['id', 'p', 'r', 'f1', 'acc', 'map'], dtype=float)
        for id in modes:
            print '\tto', id, 'dataset'
            c+=1
            preProcessId(id)
            res = eval("build"+cl+"Model(parameters, x_train, y_train, x_test, y_test, epochs)")
            if type(res) == TupleType:
                model = res[0]
                sc = res[1]
            else:
                model = res
            y_pred = eval("pred"+cl+"Model(model, x_test, y_test)")
            p, r, f1, a, m = evaluateClassifier(cl, y_pred, y_test)
            results = results.append({'id': id, 'p': p, 'r': r, 'f1': f1, 'acc': a, 'map': m}, ignore_index=True)
            print 'test', c, 'elapsed time', int(time.time())-startTime
            log.write(cl+'\t'+id+'\t'+str(c)+'\t'+str(int(time.time())-startTime)+'\n')
            log.flush()
            fsync(log.fileno())
        allResults[cl] = results
    writer = ExcelWriter(cqaDirDoc+"kerasResults_"+str(version)+".xlsx")
    for cl in whichClassifiers:
        allResults[cl].to_excel(writer,cl)
    writer.save()
    log.close()


##keras functions

def buildKerasLogRegModel(parameters, x_train, y_train, x_test, y_test, epochs):
    inputs = Input(shape=(parameters['numFeatures'],))
    output = Dense(1, activation='sigmoid')(inputs)
    kerasLogRegModel = Model(inputs, output)
    kerasLogRegModel.compile(optimizer='sgd', loss = 'binary_crossentropy', metrics=['accuracy'])
    kerasLogRegModel.optimizer.lr = 0.001
    kerasLogRegModel.fit(x=x_train, y=y_train, epochs = epochs, validation_data = (x_test, y_test),verbose=0)
    return kerasLogRegModel

def predKerasLogRegModel(kerasLogRegModel, x_test, y_test):
    y_pred = kerasLogRegModel.predict(x_test)
    return y_pred > 0.5


def buildKerasLinearRegModel(parameters, x_train, y_train, x_test, y_test, epochs):
    inputs = Input(shape=(parameters['numFeatures'],))
    output = Dense(1, activation='linear')(inputs)
    kerasLinearRegModel = Model(inputs, output)
    kerasLinearRegModel.compile(optimizer='sgd', loss = 'mse', metrics=['accuracy'])
    kerasLinearRegModel.optimizer.lr = 0.001
    kerasLinearRegModel.fit(x=x_train, y=y_train, epochs = epochs, validation_data = (x_test, y_test),verbose=0)
    return kerasLinearRegModel

def predKerasLinearRegModel(kerasLinearRegModel, x_test, y_test):
    y_pred = kerasLinearRegModel.predict(x_test)
    return y_pred > 0.5

def buildKerasSimpleFNNModel(parameters, x_train, y_train, x_test, y_test, epochs):
    inputs = Input(shape=(parameters['numFeatures'],))
    X = Dense(parameters['numFeatures']/10, activation='relu')(inputs)
    output = Dense(1, activation='sigmoid')(X)
    kerasSimpleFNNModel = Model(inputs, output)
    kerasSimpleFNNModel.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    kerasSimpleFNNModel.fit(x=x_train, y=y_train, epochs = epochs, validation_data = (x_test, y_test),verbose=0)
    return kerasSimpleFNNModel

def predKerasSimpleFNNModel(kerasSimpleFNNModel, x_test, y_test):
    y_pred = kerasSimpleFNNModel.predict(x_test)
    return y_pred > 0.5


def buildKerasDeep2FNNModel(parameters, x_train, y_train, x_test, y_test, epochs):
    return buildKerasDeepFNNModel(parameters, x_train, y_train, x_test, y_test, epochs, 2)

def buildKerasDeep3FNNModel(parameters, x_train, y_train, x_test, y_test, epochs):
    return buildKerasDeepFNNModel(parameters, x_train, y_train, x_test, y_test, epochs, 3)

def buildKerasDeep4FNNModel(parameters, x_train, y_train, x_test, y_test, epochs):
    return buildKerasDeepFNNModel(parameters, x_train, y_train, x_test, y_test, epochs, 4)

def buildKerasDeepFNNModel(parameters, x_train, y_train, x_test, y_test, epochs, numLayers):
    inputs = Input(shape=(parameters['numFeatures'],))
    X = Dense(parameters['numFeatures']/10, activation='relu')(inputs)
    X = Dropout(0.4)(X)
    for layer in range(numLayers -1):
        X = Dense(parameters['numFeatures']/20, activation='relu')(inputs)
        X = Dropout(0.3)(X)        
    output = Dense(1, activation='sigmoid')(X)
    kerasSimpleFNNModel = Model(inputs, output)
    kerasSimpleFNNModel.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    kerasSimpleFNNModel.fit(x=x_train, y=y_train, epochs = epochs, validation_data = (x_test, y_test),verbose=0)
    return kerasSimpleFNNModel

def predKerasDeep2FNNModel(kerasDeep2FNNModel, x_test, y_test):
    y_pred = kerasDeep2FNNModel.predict(x_test)
    return y_pred > 0.5

def predKerasDeep3FNNModel(kerasDeep3FNNModel, x_test, y_test):
    y_pred = kerasDeep3FNNModel.predict(x_test)
    return y_pred > 0.5

def predKerasDeep4FNNModel(kerasDeep4FNNModel, x_test, y_test):
    y_pred = kerasDeep4FNNModel.predict(x_test)
    return y_pred > 0.5


##model.add(Dense(units=64, activation = 'relu', input_dim=100))
##model.add(Dense(units=10, activation = 'softmax'))
##model.compile(loss='categorical_crossentropy',
##              optimizer = 'sgd',
##              metrics = ['accuracy'])


##main


fS = iniProcess()

globalEvaluation(clS[5:6], 10, 7, [fS[21]])

##id = fS[0]; preProcessId(id)
##epochs = 10
##print 'epochs', epochs
##print 'kerasSimpleFNNModel'
##kerasSimpleFNNModel = buildKerasSimpleFNNModel(parameters, x_train, y_train, x_test, y_test, epochs)
##y_pred = predKerasSimpleFNNModel(kerasSimpleFNNModel, x_test, y_test)
##p, r, f1, a, m = evaluateClassifier(clS[2], y_pred, y_test)
##print 'kerasDeep2FNNModel'
##kerasDeep2FNNModel = buildKerasDeep2FNNModel(parameters, x_train, y_train, x_test, y_test, epochs)
##y_pred = predKerasDeep2FNNModel(kerasDeep2FNNModel, x_test, y_test)
##p, r, f1, a, m = evaluateClassifier(clS[3], y_pred, y_test)
##print 'kerasDeep3FNNModel'
##kerasDeep3FNNModel = buildKerasDeep3FNNModel(parameters, x_train, y_train, x_test, y_test, epochs)
##y_pred = predKerasDeep3FNNModel(kerasDeep3FNNModel, x_test, y_test)
##p, r, f1, a, m = evaluateClassifier(clS[4], y_pred, y_test)
##print 'kerasDeep2FNNModel'
##kerasDeep4FNNModel = buildKerasDeep4FNNModel(parameters, x_train, y_train, x_test, y_test, epochs)
##y_pred = predKerasDeep4FNNModel(kerasDeep4FNNModel, x_test, y_test)
##p, r, f1, a, m = evaluateClassifier(clS[5], y_pred, y_test)

"""

"""
