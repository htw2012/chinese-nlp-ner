#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
from feature_utils import get_lexicals,get_onehot_lexicals_jieba
import jieba
import numpy as np

def read_by_lines(filename, word_max_len):
    sentences = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if line == "":
                continue

            line = line.decode('utf-8')
            sents = doc_to_sentence(line, word_max_len)
            sentences.extend(sents)

    return sentences

def isExists(f):
    '''
    判断文件是否存在
    :param files:
    :return:
    '''
    import os
    if not os.path.exists(f):
        return False
    return True


def load_dict(path):
    '''
    载入词典正序和逆序的映射
    :param path:
    :return:
    '''
    import pickle
    with open(path) as f:
        word_to_idx = pickle.load(f)
    return word_to_idx #, {v: k for k, v in word_to_idx.items()}


def write_file(word_to_id, dict_file):
    '''
    序列化文件
    :param word_to_id:
    :param dict_file:
    :return:
    '''
    import pickle as p
    p.dump(word_to_id, open(dict_file, 'w'))

def read_conll_file(path):
    """
    加载Conll数据格式，每行包括单词和其标签，句子由空行分割

    返回句子的列表。
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, "r", "utf8"):
        line = line.rstrip()
        # print "line", line
        if not line: # 新句子
            if len(sentence) > 0:
                if "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split(" ")
            if word:
                assert len(word) > 1, word
                sentence.append(word)

    if len(sentence) > 0:
        if "DOCSTART" not in sentence[0][0]:
            sentences.append(sentence)

    return sentences


def doc_to_sentence(doc, max_len):
    """
    文档拆分为句子（分割方式。！|？；）。
    如果超过最大长度，再以分隔符（，、／。；）进行分割
    """
    pattern1 = "。！|？；" # 通用句子分割方式
    pattern2 = "，、／。；" # 长句子分割方式

    pre_index = -1
    pattern_index = -1
    sentences = []
    sentence = []

    # 按照字进行文档拆分句子
    for i, line in enumerate(doc):
        sentence.append(line) # 加入句子

        if i - pre_index > max_len-1: # 大于最大句子长度的处理
            if pattern_index > pre_index:
                sentences.append(sentence[:pattern_index-pre_index])
                sentence = sentence[pattern_index-pre_index:]
                pre_index = pattern_index
            else:
                pre_index = i-1
                sentences.append(sentence)
                sentence = []

        else: # 小于最大长度
            # print "line-0", line[0], "pattern2", pattern2
            if line[0].encode('utf-8') in pattern2:
                pattern_index = i
                if line[0].encode('utf-8') in pattern1:
                    sentences.append(sentence)
                    sentence = []
                    pre_index = i

    if sentence:
        sentences.append(sentence)
    return sentences


def word_mapping(data, min_freq):
    '''
    构建词典
    :param data:
    :param min_freq:
    :return:
    '''

    vocab = dict() # 临时的，可能不满足
    word_to_id = dict() # 最终的词典映射
    word_id = 0 # 认为word的编号
    #line2 = "".join(data)
    for doc in data:# 每行数据
        #print "doc", doc
        line_str="".join([l[0] for l in doc])
        for line in jieba.cut(line_str):
            #print "line", line
            word = line#line[0]
            if word not in word_to_id: # 不是单字

                if word in vocab:
                    vocab[word] += 1

                    if vocab[word] >= min_freq:
                        word_to_id[word] = word_id
                        word_id += 1
                else:
                    vocab[word] = 1

    # 加入unk和pad字符
    word_to_id["<UNK>"] = word_id
    word_to_id["<PAD>"] = word_id+1
    return word_to_id, {v: k for k, v in word_to_id.items()}


def prepare_data(data, word_to_id, tag_to_id, max_words):
    '''
    根据词典将数据转化为索引格式
    :param data: 原始数据（列表的每项为句子）
    :param word_to_id: 单词到索引的映射
    :param tag_to_id: 标签到索引的映射
    :param max_words: 句子prepare_data的最大长度
    :return:
    '''

    processed_data = []
    for doc in data:#  每一行数据（空格分割）

        # 文档句子列表
        doc = doc_to_sentence(doc, max_words)
        len_doc = len(doc)

        for i, sentence in enumerate(doc):
            len_sen = len(sentence)
            str_words = []
            words = []
            tags = []
            for line in sentence:
                # 规范化
                word = line[0].lower()
                str_words.append(word)
                words.append(word_to_id[word] if word in word_to_id else word_to_id["<UNK>"])
                tags.append(tag_to_id[line[-1]])

            # 尾部补齐
            words += [word_to_id["<PAD>"]] * (max_words-len_sen)
            tags += [tag_to_id["O"]] * (max_words-len_sen)

            # BIES打标签
            features = get_onehot_lexicals_jieba(max_words, str_words)


            processed_data.append({"str_line": str_words,
                                   "words": words, # 补齐后
                                   "tags": tags,
                                   "len": len_sen, # real sentence len
                                   "features": features,
                                   "end_of_doc": i == len_doc-1})
    return processed_data


def load_data(params, tag_to_id, only_use_test=False, specific_file = None):
    '''
    载入数据
    :param params:
    :param tag_to_id:
    :return:
    '''
    if only_use_test:
        filename = specific_file if specific_file else params.test_file
        test_file = read_conll_file(filename)
        print "only using test_code files ......"

        print "dict building......"
        if not isExists(params.dict_file):
            print "build dict starting..."
            train_file = read_conll_file(params.train_file)
            word_to_id, _ = word_mapping(train_file, params.min_freq)
            write_file(word_to_id, params.dict_file)
        else:
            print "build dict from pickle..."
            word_to_id = load_dict(params.dict_file)
        test_data = prepare_data(test_file, word_to_id, tag_to_id, params.word_max_len)
        return word_to_id, {v: k for k, v in tag_to_id.items()}, None, None, test_data

    print "file read staring.....", params.train_file
    train_file = read_conll_file(params.train_file)
    print "result  train_file.....", len(train_file), "first", train_file[0], "type", type(train_file)
    dev_file = read_conll_file(params.dev_file)
    test_file = read_conll_file(params.test_file)
    print "file read  ending......"

    if not isExists(params.dict_file):
        print "build dict starting..."
        word_to_id, _ = word_mapping(train_file, params.min_freq)
        write_file(word_to_id, params.dict_file)
    else:
        print "build dict from pickle..."
        word_to_id = load_dict(params.dict_file)

    print "word_to_id-size", len(word_to_id)
    # print "id_to_word-size", len(id_to_word)

    train_data = prepare_data(train_file, word_to_id, tag_to_id, params.word_max_len)
    dev_data = prepare_data(dev_file, word_to_id, tag_to_id, params.word_max_len)
    test_data = prepare_data(test_file, word_to_id, tag_to_id, params.word_max_len)
    print(len(train_file))
    print(len(train_data))

    return word_to_id, {v: k for k, v in tag_to_id.items()}, train_data, dev_data, test_data


if __name__ == '__main__':
    import train as tr
    params = tr.FLAGS
    tag_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}

    id_to_word, id_to_tag, train_data, dev_data, test_data = load_data(params, tag_to_id)

    print "id_to_word"
