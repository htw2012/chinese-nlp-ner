#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

    # load data
    id_to_word, id_to_tag, train_data, dev_data, test_data = load_data(FLAGS, tag_to_id)
    train_manager = BatchManager(train_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)
    test_manager = BatchManager(test_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(sess, id_to_word, id_to_tag)
        loss = 0
        best_test_f1 = 0
        steps_per_epoch = len(train_data) // FLAGS.batch_size + 1
        for _ in range(FLAGS.max_epoch):
            iteration = (model.global_step.eval()) // steps_per_epoch + 1
            train_manager.shuffle()
            for batch in train_manager.iter_batch():
                global_step = model.global_step.eval()
                step = global_step % steps_per_epoch
                batch_loss = model.run_step(sess, True, batch)
                loss += batch_loss / FLAGS.steps_per_checkpoint
                if global_step % FLAGS.steps_per_checkpoint == 0:
                    model.logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}"
                                      .format(iteration,
                                              step,
                                              steps_per_epoch,
                                              loss))
                    loss = 0

            model.logger.info("validating ner")
            ner_results = model.predict(sess, dev_manager)
            eval_lines = conll_eval(ner_results, FLAGS.result_path)
            for line in eval_lines:
                model.logger.info(line)
            test_f1 = float(eval_lines[1].strip().split()[-1])
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                model.logger.info("new best f1 score:{:>.3f}".format(test_f1))
                model.logger.info("saving model ...")
                checkpoint_path = os.path.join(FLAGS.model_path, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

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
