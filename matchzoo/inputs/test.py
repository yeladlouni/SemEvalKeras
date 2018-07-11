from matchzoo.inputs.point_generator import PointGenerator

config = {}
config['data1'] = "./data/SemEval/corpus_preprocessed.txt"
config['data2'] = "./data/SemEval/corpus_preprocessed.txt"
config['relation_file'] = "./data/SemEval/corpus_preprocessed.txt"
config['batch_size'] = 1
config['text1_maxlen'] = 5
config['text2_maxlen'] = 5
config['vocab_size'] = 88742
config['target_mode'] = "ranking"
config['phase'] = 'TRAIN'

generator = PointGenerator(config)

generator.get_batch()