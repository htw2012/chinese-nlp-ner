#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "模型的配置信息"
author= "huangtw"
mtime= 2017-07-03
"""

types = "sighan"#XX
feature_dim = 264#4 # 0 #, "dimension of extra features, 0 for not used")

corpus_type = "sighan"#"sighan_rule"#" "rule_gen"#"sighan"#

# path for log, model and result
log_path = "./logs/"+types # "path for log files"
model_path = "./weights/"+types  # "path to save model"
result_path = "./results/"+types # "path to save result"

train_file = "./data/%s/%s.train"%(corpus_type, corpus_type) # , "path for train data")
dev_file = "./data/%s/%s.dev"%(corpus_type, corpus_type) #, "path for valid data")
test_file = "./data/%s/%s.test"%(corpus_type, corpus_type) # , "path for test data")

# config for model
lower = True #, "True for lowercase all characters")
pre_emb = "./embedding/word2vec_sg_py2.pkl" #"path for pre-trained embedding, False for randomly initialize")
min_freq = 2
# 字典文件
dict_file = "./data/%s/dict_file.pickle"%(corpus_type)

# tagger
tag_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}

# using for 100
word_max_len = 100 # "maximum words in a sentence")
word_dim = 100 # "dimension of char embedding")
word_hidden_dim = 150 # , "dimension of word LSTM hidden units")

# config for training process
dropout = 0.5 #  "dropout rate")
clip = 5 #  "gradient to clip")
lr = 0.001 # "initial learning rate")
max_epoch = 150 # "maximum training epochs")
batch_size = 20 # "num of sentences per batch")
steps_per_checkpoint = 100 #"steps per checkpoint")
valid_batch_size = 100 #, "num of sentences per batch")
