#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "特征获取工具类"
author= "huangtw"
mtime= 2017-08-24
"""
import jieba
import jieba.posseg as pseg
import json
import numpy as np

tag_idx = {}

postag_id = {u'en': 11, u'vd': 53, u'ad': 1, u'ag': 2, u'vg': 54, u'nrfg': 28, u'zg': 63, u'vn': 56, u'an': 3, u'vq': 57, u'in': 16, u'ud': 46, u'rr': 40, u'p': 34, u'ln': 21, u'mq': 24, u'rz': 41, u'ng': 26, u'nz': 32, u'rg': 39, u'vi': 55, u'qe': 36, u'nr': 27, u'ns': 30, u'nt': 31, u'bg': 5, u'qg': 37, u'f': 12, u'df': 8, u'dg': 9, u'uj': 48, u'c': 6, u'yg': 61, u'jn': 18, u'tg': 44, u'j': 17, u'nrt': 29, u'a': 0, u'mg': 23, u'b': 4, u'e': 10, u'd': 7, u'g': 13, u'uz': 51, u'i': 15, u'h': 14, u'k': 19, u'uv': 50, u'm': 22, u'l': 20, u'o': 33, u'n': 25, u'q': 35, u'ul': 49, u's': 42, u'r': 38, u'u': 45, u't': 43, u'w': 58, u'v': 52, u'y': 60, u'x': 59, u'ug': 47, u'z': 62};


def get_onehot_lexicals_jieba(max_words, str_words):
    '''
    获得句子的词法信息
    :param max_words:
    :param str_words:
    :return:
    '''
    term_num_tags = 1
    pos_num_tags = 64

    features = np.zeros([max_words, (term_num_tags+pos_num_tags+1)*4], dtype=np.float32)  # 写死了
    index = 0
    line = "".join(str_words)
    if(len(line)>max_words):
        print "max-line", line, "size", len(line)
    # BIES tags（结巴分词加入外在的特征）
    for word in jieba.cut(line):  # 直接分词
        len_word = len(word)
        if len_word == 1:
            features[index, 0] = 1  # S
            index += 1
        else:
            features[index, 1] = 1  # B
            index += 1
            for i_ in range(len_word - 2):  # I
                features[index, 2] = 1
                index += 1

            features[index, 3] = 1  # E
            index += 1

    index = 0 
    # 加入pos信息
    for word, flag in pseg.cut(line):  # 直接分词
        if(postag_id.has_key(flag)):
            base = postag_id[flag]+1 #
        else:
            base = 65
       # print "word", word, "tag", flag, "base", base
        len_word = len(word)
        if len_word == 1:
            features[index, base*4 + 0] = 1  # S
            index += 1
        else:
            features[index, base*4 + 1] = 1  # B
            index += 1
            for i_ in range(len_word - 2):  # I
                features[index, base*4 + 2] = 1
                index += 1

            features[index, base*4 + 3] = 1  # E
            index += 1

    return features



if __name__ == '__main__':
    max_words = 20
    text = "张三去五星小学去办事，函授课程，祝愿愉快的办好事情，导航去北京的地铁知春路到南京的上海路坐公交到澳大利亚需要怎么走耍耍"
    text = ""
    text = "张三去五星小学去打麻将12a"
    text = "帮我查询一下三里屯附近的西餐厅"
    text = "桂林《叮叮糖》简介"
    # features = get_lexicals(max_words, text)

    # print "features", features
    # for word in jieba.cut("".join(text)):
    #     print "jieba", word

    # ictclas
    # import jieba.posseg as pseg
    # words = pseg.cut("我爱北京天安门")
    # for word, flag in words:
    #      print('%s %s' % (word, flag))
    str_words = "桂林《叮叮糖》简介"
    features = get_onehot_lexicals_jieba(max_words, str_words)
    # np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print "features", features
    print "shape", features.shape
