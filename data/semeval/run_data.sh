#!/bin/bash
# help, dos2unix file

# download the dataset if not already done
if [ ! -d ./Task3 ]; then
    # train and dev datasets
    [ -f ./semeval2016-task3-cqa-arabic-md-train-v1.3.zip ] && echo "File already exists." || wget http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-arabic-md-train-v1.3.zip
    unzip semeval2016-task3-cqa-arabic-md-train-v1.3.zip "SemEval2016-Task3-CQA-Arabic-MD-train-v1.3/*.zip"
    mkdir Task3 && cd SemEval2016-Task3-CQA-Arabic-MD-train-v1.3 && unzip \*.zip && cp *.xml ../Task3
    cd .. && rm -rf SemEval2016-Task3-CQA-Arabic-MD-train-v1.3
    
    # 2017 test dataset
    [ -f ./semeval2017_task3_test.zip ] && echo "File already exists." || wget http://alt.qcri.org/semeval2017/task3/data/uploads/semeval2017_task3_test.zip
    unzip semeval2017_task3_test.zip
    gunzip ./SemEval2017_task3_test/Arabic/SemEval2017-Task3-CQA-MD-test.xml.gz && cp ./SemEval2017_task3_test/Arabic/SemEval2017-Task3-CQA-MD-test.xml ./Task3
    rm -rf SemEval2017_task3_test && rm -rf __MACOSX

    # 2016 test dataset
    [ -f ./semeval2016_task3_test.zip ] && echo "File already exists." || wget http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016_task3_test.zip
    unzip semeval2016_task3_test.zip
    gunzip ./SemEval2016_task3_test/Arabic/SemEval2016-Task3-CQA-MD-test.xml.gz && cp ./SemEval2017_task3_test/Arabic/SemEval2016-Task3-CQA-MD-test.xml ./Task3
    rm -rf SemEval2016_task3_test && rm -rf __MACOSX
    
fi

# parse the xml files into flat files
python parse_xml.py

# prepare data
python prepare_data.py

# generate word embedding
python gen_w2v.py webteb.d128.txt word_dict.txt embed_word2vec_d128
python norm_embed.py embed_word2vec_d128 embed_word2vec_d128_norm

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
python gen_hist4drmm.py 60
python gen_binsum4anmm.py 20 # the default number of bin is 20

echo "Done ..."
