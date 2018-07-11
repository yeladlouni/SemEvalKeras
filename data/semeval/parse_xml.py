# coding: utf-8
"""
Parse train & dev datasets from XML files provided by SemEval
"""
from __future__ import print_function

import xml.etree.ElementTree as ET

if __name__ == '__main__':
    basedir = './Task3/'
    in_corpfile = [basedir + 'SemEval2016-Task3-CQA-MD-train.xml', basedir + 'SemEval2016-Task3-CQA-MD-test.xml', basedir + 'SemEval2017-Task3-CQA-MD-test.xml']
    out_corpfile = [basedir + 'SemEval_train.txt', basedir + 'SemEval_dev.txt', basedir + 'SemEval_test.txt']

    for i in range(len(in_corpfile)):
        with open(out_corpfile[i], 'w', encoding='utf-8') as fout:
            tree = ET.parse(in_corpfile[i])
            root = tree.getroot()

            for Question in root:
                QID = int(Question.get('QID'))
                Qtext = Question.find('Qtext').text
                
                for QApair in Question.iter('QApair'): 
                    QAID = int(QApair.get('QAID'))
                    QArel = 0 if QApair.get('QArel') == 'I' else 1
                    QAconf = float(0 if QApair.get('QAconf') == None else QApair.get('QAconf'))
                    QAquestion = QApair.find('QAquestion').text
                    QAanswer = QApair.find('QAanswer').text
                    
                    print(str(QArel) + '\t' + Qtext + '\t' + QAquestion + '\t' + 'Q'+str(QID) + '\t' + 'QA'+str(QAID), file=fout)

