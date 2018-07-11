#!/usr/bin/env bash
#-*- coding: utf-8 -*-
#@Filename : fasttext
#@Date : 2018-06-20-07-30
#@Poject: SemEval2
#@AUTHOR : Yassine EL ADLOUNI


python gen_w2v.py fasttext-webteb.txt vocab.txt embed_word2vec_d128
python norm_embed.py embed_word2vec_d128 embed_word2vec_d128_norm