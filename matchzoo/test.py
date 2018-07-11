from inputs.pair_generator import PairGenerator
from inputs.list_generator import ListGenerator
from inputs.point_generator import PointGenerator
from utils import *
import numpy as np
from keras import backend as K

config = {}
config['text1_corpus'] = "./data/SemEval/corpus_preprocessed.txt"
config['text2_corpus'] = "./data/SemEval/corpus_preprocessed.txt"
config['phase'] = 'TRAIN'
config['relation_file'] = "./data/SemEval/relation_train.txt"
config['batch_size'] = 1
config['text1_maxlen'] = 100
config['text2_maxlen'] = 100
config['vocab_size'] = 88742
config['target_mode'] = "classification"
config['use_iter'] = False
config['use_dpool'] = False
config['num_iters'] = 1

dataset = {}


datapath = config['text1_corpus']
if datapath not in dataset:
    dataset[datapath], _ = read_data(datapath)

datapath = config['text2_corpus']
if datapath not in dataset:
    dataset[datapath], _ = read_data(datapath)

config['data1'] = dataset[config['text1_corpus']]
config['data2'] = dataset[config['text2_corpus']]

config['query_per_iter'] =  1
config['batch_per_iter'] =  1
config['batch_size'] =  3
config['batch_list'] = 1
config['target_mode'] = 'classification'
config['class_num'] = 2

generator = PointGenerator(config)

genfun = generator.get_batch_generator()
i=0
np.random.seed(123)
for input_data, y_true in genfun:
    
    print(y_true)
    break

