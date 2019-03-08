# coding: utf-8
import re
import numpy as np
import pandas as pd
import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

Lem = WordNetLemmatizer()
def invert_dict(d):
    return {v:k for k,v in iter(d.items())}


def load_lm_dataset(fname):
    sents = []
    cnt = 0
    with open(fname) as f:
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            items = line.strip().split('\t')
            sents.append(items[0].split())
    return sents


def load_np_dataset(fname):
    sents = []
    cnt = 0
    with open(fname) as f:
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            items = line.strip().split('\t')
            verb_idx = int(items[2])
            verb_pos = items[3]
            sent = [verb_pos] + items[0].split()[:verb_idx]
            sents.append(sent)
    return sents


def load_lm_np_dataset(fname):
    sents = []
    cnt = 0
    with open(fname) as f:
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            items = line.strip().split('\t')
            verb_idx = int(items[2])
            verb = items[4]
            inf_verb = items[5]
            sent = items[0].split()[:verb_idx] + [verb, inf_verb]
            sents.append(sent)
    return sents


def load_grammar_dataset(fname):
    sents = []
    cnt = 0
    cnt_ungram=0
    f=open(fname)
    file_contents = f.read()
    f_c=file_contents.split('\n')

    with open(fname) as f:
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            items = line.strip().split('\t')
            if random.random()>0.4 and cnt_ungram<(len(f_c)-1)/2:
                verb_idx = int(items[2])
                swap_verb= items[-1]
                sent=items[0].split()
                sent[verb_idx]=swap_verb
                
                sent1=['cat_ungrammatical']+sent
                cnt_ungram+=1
            else:
                sent=items[0].split()
                
                sent1=['cat_grammatical']+sent
               
            sents.append(sent1)
    #print(sents)        
    return cnt_ungram,len(f_c),sents


def load_noun_dataset(fname):
    sents = []
    cnt = 0
    noun_set=[synset.lemmas()[0].name() for synset in list(wn.all_synsets(wn.NOUN))]
   
    with open(fname) as f:
        for line in f:
            new_sent=''
            if cnt == 0:
                cnt += 1
                continue
            items = line.strip().split('\t')
            verb_idx = int(items[2])
            verb_pos = items[3]
            
            sent = [verb_pos]
            for item in items[0].split()[:verb_idx]:
                
                if Lem.lemmatize(item) in noun_set:
                    sent.append(item)
            
            if(len(sent)>1):
                sents.append(sent)
    return sents


def pad_sequence(seq, left=1, right=1):
    return left*["<s>"] + seq + right*["</s>"]


# For RNN
# just convert each sentence to a list of indices
# after padding each with <s> ... </s> tokens
def seq_to_indices(words, word_to_num):
    return np.array([word_to_num[w] for w in words])


def docs_to_indices(sents, word_to_num, pad_left=1, pad_right=1):
    sents = [pad_sequence(s, pad_left, pad_right) for s in sents]
    sents_idx = []
    for sent in sents:
        words = [w if w in word_to_num else 'UNK' for w in sent]
        sents_idx.append(seq_to_indices(words, word_to_num))

    # return as numpy array for fancier slicing
    return np.array(sents_idx, dtype=object)


def offset_seq(seq):
    return seq[:-1], seq[1:]


def offset_np(seq):
    return seq[1:], [seq[0]]


def offset_lm_np(seq):
    return seq[:-2], [seq[-2], seq[-1]]


def seqs_to_lmXY(seqs):
    X, Y = zip(*[offset_seq(s) for s in seqs])
    return np.array(X, dtype=object), np.array(Y, dtype=object)


def seqs_to_npXY(seqs):
    X, Y = zip(*[offset_np(s) for s in seqs])
    return np.array(X, dtype=object), np.array(Y, dtype=object)


def seqs_to_lmnpXY(seqs):
    X, Y = zip(*[offset_lm_np(s) for s in seqs])
    return np.array(X, dtype=object), np.array(Y, dtype=object)
