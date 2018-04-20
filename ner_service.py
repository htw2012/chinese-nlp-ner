#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from model import Model
from data_loader import load_dict, write_file, isExists,prepare_data,read_conll_file, word_mapping
from data_utils import BatchManager

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import config_ner as FLAGS


def create_model(session, word_to_id, id_to_tag):
    # create model, reuse parameters if exists
    model = Model("tagger", word_to_id, id_to_tag, FLAGS)
    print "FLAGS.model_path", FLAGS.model_path
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        model.logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def get_test_data2(text):
    """
    获得测试数据
    """
    sentences = []
    sentence = []
    for word in text.decode('utf-8'):
        sentence.append([word, u'O']) # default is O, for test

        if len(sentence) == FLAGS.word_max_len:
            # print "append", sentence
            sentences.append(sentence)
            sentence = []

    if sentence:# 剩余部分
        # print "last",sentence
        sentences.append(sentence)
    print "sentence", sentence
    return sentences


def create_instance():
    tag_to_id = FLAGS.tag_to_id
    id_to_tag = {v: k for k, v in tag_to_id.items()}

    # 字典生成
    print "dict building......"
    if not isExists(FLAGS.dict_file):
        print "build dict starting..."
        train_file = read_conll_file(FLAGS.train_file)
        word_to_id, _ = word_mapping(train_file, FLAGS.min_freq)
        write_file(word_to_id, FLAGS.dict_file)
    else:
        print "build dict from pickle..."
        word_to_id = load_dict(FLAGS.dict_file)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with sess.as_default():
        print "begin for create model..."
        model = create_model(sess, word_to_id, id_to_tag) # just struct

        # load model
        model.logger.info("testing ner")
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

    return word_to_id, tag_to_id, id_to_tag, sess, model


def get_batch_manager(id_to_tag, tag_to_id, text, word_to_id):
    test_file = get_test_data2(text)
    test_data = prepare_data(test_file, word_to_id, tag_to_id, FLAGS.word_max_len)
    # load data,迭代器
    test_manager = BatchManager(test_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)
    return test_manager




