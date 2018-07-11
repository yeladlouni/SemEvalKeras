# -*- coding: utf-8 -*-
# !/pkg/ldc/bin/python2.7
#-----------------------------------------------------------------------------
# Name:        runScikitLearnExperiments.py
#
# Author:      Horacio
#
# Created:     2018/01/18
# Run ScikitLearn experiments following Geron book
#-----------------------------------------------------------------------------


import string
import re
from types import IntType,ListType,StringType,UnicodeType,BooleanType,TupleType
from os import listdir
import itertools
import numpy as np
from scipy import stats
from sklearn import svm, linear_model, cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import pickle
import gzip
import os
import pandas as pd
from pandas import DataFrame, ExcelWriter

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
clS = ['SGD','LR','Ridge','Lasso','EN','LogReg','SoftMax']

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
    num_classes = 1
    print num_classes, 'classes'
    

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


def buildSGDClassifier(x_train, y_train):
    """
    Stochastic Gradient Descent Classifier
    """
    from sklearn.linear_model import SGDClassifier
    sgd_cl = SGDClassifier(random_state=42)
    sgd_cl.fit(x_train, y_train)
    sc = cross_val_score(sgd_cl, x_train, y_train, cv = 3, scoring = "accuracy")
    return sgd_cl, np.average(sc)
                
def buildLRClassifier(x_train, y_train):
    """
    Linear Regression Classifier
    """
    from sklearn.linear_model import LinearRegression
    lR_cl = LinearRegression()
    lR_cl.fit(x_train, y_train)
    return lR_cl
                
def buildRidgeClassifier(x_train, y_train):
    """
    Regularized Linear Models: Ridge Regression
    """
    from sklearn.linear_model import Ridge
    ridge_cl = Ridge(alpha = 1, solver = 'cholesky')
    ridge_cl.fit(x_train, y_train)
    return ridge_cl
                
def buildLassoClassifier(x_train, y_train):
    """
    Regularized Linear Models: Lasso
    """
    from sklearn.linear_model import Lasso
    lasso_cl = Lasso(alpha = 0.1)
    lasso_cl.fit(x_train, y_train)
    return lasso_cl
                
def buildENClassifier(x_train, y_train):
    """
    Regularized Linear Models: Elastic Net
    """
    from sklearn.linear_model import ElasticNet
    en_cl = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
    en_cl.fit(x_train, y_train)
    return en_cl
                
def buildLogRegClassifier(x_train, y_train):
    """
    Regularized Linear Models: Logistic Regression
    """
    from sklearn.linear_model import LogisticRegression
    logReg_cl = LogisticRegression()
    logReg_cl.fit(x_train, y_train)
    return logReg_cl
                
def buildSoftMaxClassifier(x_train, y_train):
    """
    SoftMax
    """
    from sklearn.linear_model import LogisticRegression
    softMax_cl = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", C=2)
    softMax_cl.fit(x_train, y_train)
    return softMax_cl
                
def predSGDClassifier(sgd_cl, x_test, y_test):
    y_pred = sgd_cl.predict(x_test)
    return y_pred

def predLRClassifier(lR_cl, x_test, y_test):
    y_pred = lR_cl.predict(x_test)
    y_pred = y_pred>0
    return y_pred

def predRidgeClassifier(ridge_cl, x_test, y_test):
    y_pred = ridge_cl.predict(x_test)
    y_pred = y_pred>0
    return y_pred

def predLassoClassifier(lasso_cl, x_test, y_test):
    y_pred = lasso_cl.predict(x_test)
    y_pred = y_pred>0
    return y_pred

def predENClassifier(en_cl, x_test, y_test):
    y_pred = en_cl.predict(x_test)
    y_pred = y_pred>0
    return y_pred

def predLogRegClassifier(logReg_cl, x_test, y_test):
    y_pred = logReg_cl.predict(x_test)
    return y_pred

def predSoftMaxClassifier(softMax_cl, x_test, y_test):
    y_pred = softMax_cl.predict(x_test)
    return y_pred

def evaluateClassifier(cl, y_pred, y_test):
    p = precision_score(y_pred, y_test)
    r = recall_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    a = accuracy_score(y_pred, y_test)
    m = average_precision_score(y_pred, y_test)
    print 'precision', p
    print 'recall', r
    print 'f1', f1
    print 'accuracy', a
    print 'map', m
    return p, r, f1, a, m

def globalEvaluation(whichClassifiers, version = 0):
    global fS, allResults
    allResults = {}
    for cl in whichClassifiers:
        print 'applying', cl, 'classifier'
        results = DataFrame(columns=['id', 'p', 'r', 'f1', 'acc', 'map'], dtype=float)
        for id in fS:
            print '\tto', id, 'dataset'
            preProcessId(id)
            res = eval("build"+cl+"Classifier(x_train, y_train)")
            if type(res) == TupleType:
                model = res[0]
                sc = res[1]
            else:
                model = res
            y_pred = eval("pred"+cl+"Classifier(model, x_test, y_test)")
            p, r, f1, a, m = evaluateClassifier(cl, y_pred, y_test)
            results = results.append({'id': id, 'p': p, 'r': r, 'f1': f1, 'acc': a, 'map': m}, ignore_index=True)
        print results
        allResults[cl] = results
    writer = ExcelWriter(cqaDirDoc+"scikitLearnResults_"+str(version)+".xlsx")
    for cl in whichClassifiers:
        allResults[cl].to_excel(writer,cl)
    writer.save()

##main

fS = iniProcess()
globalEvaluation(clS[2:], 1)
    
##id = fS[0]; preProcessId(id)
##softMax_cl = buildSoftMaxClassifier(x_train, y_train); print softMax_cl; p, r, f1, a, m = evaluateClassifier(softMax_cl, y_pred, y_test)
##
##lR_cl = buildLRClassifier(x_train, y_train)
##print lR_cl
##y_pred = predLRClassifier(lR_cl, x_test, y_test)
##evaluateClassifier(lR_cl, y_pred, y_test)
                    
                         

"""

"""
