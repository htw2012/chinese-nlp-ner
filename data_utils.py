#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import random
import logging
import numpy as np
import jieba


def get_lexical_feature(max_words, str_words):
    '''
    获得词性的特征
    :param max_words:
    :param str_words:
    :return:
    '''
    features = np.zeros([max_words, 4], dtype=np.float32) # 写死了
    index = 0
    # BIES tags（结巴分词加入外在的特征）
    for word in jieba.cut("".join(str_words)):  # 直接分词

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
    return features


def load_word2vec(path, id_to_vec):
    with open(path, "rb") as f:
        word_vec = pickle.load(f)
        word2vec = []
        for i, word in id_to_vec.items():
            if word in word_vec:
                word2vec.append(word_vec[word])
            else:
                vec = [0.0 for _ in range(100)]
                word2vec.append(vec)
    return word2vec


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join("./log", name + ".log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def conll_eval(results, path):
    script_file = "./conlleval"
    output_file = os.path.join(path, "XX.utf8")
    result_file = os.path.join(path, "XX.utf8")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block[0]:
                to_write.append(line + "\n")
            if block[1]:
                to_write.append("\n")
        # f.writelines(to_write)
        for line in to_write:
            f.write(line)
        # f.writelines([str(line) + "\n" for line in to_write])
    os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
    eval_lines = []
    with open(result_file) as f:
        for line in f:
            eval_lines.append(line.strip())
    return eval_lines


def calculate_accuracy(labels, paths, lengths):
    # calculate token level accuracy, return correct tag numbers and total tag numbers
    total = 0
    correct = 0
    for label, path, length in zip(labels, paths, lengths):
        gold = label[length]
        correct += np.sum(np.equal(gold, path))
        total += length
    return correct, total


class BatchManager(object):

    def __init__(self, data, num_tag, max_len, batch_size):
        self.data = data
        self.numbatch = len(self.data) // batch_size
        # print "numbatch", self.numbatch, "batch_size", batch_size, "len", len(self.data)
        self.batch_size = batch_size
        self.batch_index = 0
        self.len_data = len(data)
        self.num_tag = num_tag

    @staticmethod
    def unpack(data):
        words = []
        tags = []
        lengths = []
        features = []
        str_lines = []
        end_of_doc = []
        for item in data:
            # print "sent-len", item["len"]
            if item["len"] < 0:
                continue
            words.append(item["words"])
            tags.append(item["tags"])
            lengths.append(item["len"])

            features.append(item["features"])

            str_lines.append(item["str_line"])
            end_of_doc.append(item["end_of_doc"])

        return {"words": words,
                "tags": tags,
                "len": lengths,
                "features": features,
                "str_lines": str_lines,
                "end_of_doc": end_of_doc}

    def shuffle(self):
        random.shuffle(self.data)

    def iter_batch(self):
        for i in range(self.numbatch+1): # fix last part
            # print "doing batch", i
            if i == self.numbatch:
                data = self.data[i*self.batch_size:]# 后面部分

            else:
                data = self.data[i*self.batch_size:(i+1)*self.batch_size]
                # print "data-len1", len(data)
            yield self.unpack(data)

