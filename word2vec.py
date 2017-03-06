#!/usr/sbin/python3
#-*- encoding:utf-8 -*-

from gensim.models.word2vec import *

model_file = 'word2vec.model'

with open('user_prefs.txt') as f:
    prefs_str = ''.join(f.readlines())

# {'andy': {'霍乱时期的爱情': 1},...}
def read_prefs(prefs_str):
    prefs = {}
    for line in prefs_str.split('\n'):
        parts = line.rstrip().split()
        if len(parts) == 2:
            userId, itemId = parts
            prefs.setdefault(userId, {})
            prefs[userId].update({itemId:1})
    return prefs

prefs = read_prefs(prefs_str)

def sents_from_prefs(prefs):
    sents = []
    for v in prefs.values():
        sent = ''
        for b in v.keys():
            sent += ' ' + b.replace(' ', '')
        sents.append(sent)
    return sents

def flatMap(vocab):
    ret = []
    for i in vocab:
        if type(i) == type('a'):
            ret.append(i)
        elif type(i) == type([]):
            for j in i:
                ret.append(j)
    return ret

def calc_item_cf():
    sents = sents_from_prefs(prefs) # LineSentence('README.md')
    vocab = [s.split() for s in sents]
    model = Word2Vec(vocab, size=100, window=5, min_count=1, workers=4)
    model.save_word2vec_format(model_file, binary=False)
    model = Word2Vec.load_word2vec_format(model_file, binary=False)

    print('基于书籍的 word2vec 协同过滤推荐')
    for item in flatMap(vocab):
        print('\n根据 %s 推荐：' % item)
        for item_score in model.most_similar(positive=[item]):
            item, score = item_score
            print('\t%s %.2f' % (item, score))

calc_item_cf()