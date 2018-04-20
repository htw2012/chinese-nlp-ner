#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "模型的评估，需要实际的标签信息"
author= "huangtw"
mtime= 2017-06-30
"""

import os
import tensorflow as tf
from model import Model
from data_loader import load_data
from data_utils import BatchManager, conll_eval

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import config_ner as FLAGS


def create_model(session, word_to_id, id_to_tag):
    # create model, reuse parameters if exists
    model = Model("tagger", word_to_id, id_to_tag, FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        model.logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def main(_):
    if not os.path.isdir(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    if not os.path.isdir(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)
    if not os.path.isdir(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)

    tag_to_id = FLAGS.tag_to_id

    # specific_file = "data/mor-test/test_set.mor"
    specific_file = "../mor_v1_addr.test"#"../addr_all.test"#""#"data/mor-test_code/mor_iter_v1/mor_person_label_v2.txt"#FLAGS.test_file#""data/rule_gen/rule_gen.test_code"#"data/sighan/sighan.test_code"#
    # load data
    id_to_word, id_to_tag, _, _, test_data = load_data(FLAGS, tag_to_id, only_use_test=True, specific_file=specific_file)
    test_manager = BatchManager(test_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(sess, id_to_word, id_to_tag)
        # test model
        model.logger.info("testing ner")
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        ner_results = model.predict(sess, test_manager)
        eval_lines = conll_eval(ner_results, FLAGS.result_path)
        for line in eval_lines:
            model.logger.info(line)


if __name__ == "__main__":
    tf.app.run(main)
