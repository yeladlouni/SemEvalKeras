#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : convert
#@Date : 2018-06-20-07-15
#@Poject: SemEval2
#@AUTHOR : Yassine EL ADLOUNI


with open('fasttext.webteb.100d.vec', 'r') as f:
    next(f)
    with open('fasttext-webteb.txt', 'w') as f1:
        with open('vocab.txt', 'w') as f2:
            for idx, line in enumerate(f):
                parsed = line.rstrip().split(' ')
                w = parsed[0]
                vec = ' '.join(parsed[1:])
                f1.write(vec + '\n')
                f2.write(w + ' ' + str(idx) + '\n')