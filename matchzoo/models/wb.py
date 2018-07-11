from __future__ import print_function 
from __future__ import division 

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET

import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FTRL

from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def read_xml(path):

    tree = ET.parse(path)
    root = tree.getroot()
    
    df = pd.DataFrame(columns=['QID', 'QAID'], dtype=int)

    for Question in root:
        QID = int(Question.get('QID'))
        Qtext = Question.find('Qtext').text
        
        for QApair in Question.iter('QApair'): 
            QAID = int(QApair.get('QAID'))
            QArel = 0 if QApair.get('QArel') == 'I' else 1
            QAconf = 0.0 if QApair.get('QAconf') == None else float(QApair.get('QAconf'))
            QAquestion = QApair.find('QAquestion').text
            QAanswer = QApair.find('QAanswer').text
            
            df = df.append({'QID': QID,
                            'QAID': QAID,
                            'Qtext': Qtext,
                            'QAquestion': QAquestion,
                            'QAanswer': QAanswer,
                            'QArel': QArel,
                            'QAconf': QAconf}, ignore_index=True) 
        
    #df.set_index(['QID', 'QAID'], inplace=True)
    return df

def normalize_text(text):
    stemmer = ISRIStemmer()
    return ' '.join([stemmer.stem(w) for w in wordpunct_tokenize(text)])


def main():
    df_train = read_xml('/media/yassine/Data/SemEval2/data/semeval/Task3/SemEval2016-Task3-CQA-MD-train.xml')
    print(len(set(df_train['QID'])))
    print(len(df_train['QAID']))
    #df_test = read_xml('/media/yassine/Data/SemEval2/data/semeval/Task3/SemEval2017-Task3-CQA-MD-test.xml')
 
    ''' print('train dataset: ', df_train.shape)
    print('test dataset: ', df_test.shape)

    merge = pd.concat([df_train, df_test])


    print('merge: ', merge.shape)

    nrow = df_train.shape[0]

    del df_train
    del df_test
    
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [0.5, -1.0],
                                                                  "hash_size": 2 ** 23, "norm": 'l2', "tf": 'log',
                                                                  "idf": 20.0}))
    wb.dictionary_freeze= True

    X = merge['Qtext'] + ' ' + merge['QAquestion'] + ' ' + merge['QAanswer']

    X = wb.fit_transform(X)
    y = merge['QArel']

    #X_query = X_query[:, np.array(np.clip(X_query.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    #X_question = X_question[:, np.array(np.clip(X_question.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    #X_answer = X_answer[:, np.array(np.clip(X_answer.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    X = X[:, np.array(np.clip(X.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    del(wb)

    print('X: ', X.shape)
    #print('X_question: ', X_question.shape)
    #print('X_answer: ', X_answer.shape)

    print('y: ', y.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(X[:nrow], y[:nrow], test_size=0.2)

    clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=10)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)
    y_pred = [1 if score >= 0 else 1 for score in preds]

    accuracy = accuracy_score(y_valid, y_pred)
    print(accuracy)
 '''
if __name__ == '__main__':
    main()
