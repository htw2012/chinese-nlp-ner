#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2017-06-27
"""
from flask import Flask,request
import ner_service as n_service
import json
import numpy as np

app = Flask(__name__)
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
dict={"LOC":"Location","PER":"Person","ORG":"Organization"}

class Item(object):

    # 定义静态变量实例
    def __init__(self):
        self.start = -1
        self.end = -1
        self.text = ""
        self.type = ""
        self.score = 0

    def to_string(self):
        dict = {}
        dict["start"] = self.start
        dict["end"] = self.end
        dict["text"] = self.text
        #print "val-repr-text", repr(dict["text"])
        dict["type"] = self.type
        dict["score"] = self.score

        str = json.dumps(dict, False, False, indent=4)#.encode("utf-8")
        return str


class Singleton(object):
    # 定义静态变量实例
    __singleton = None
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        if Singleton.__singleton is None:
            word_to_id, tag_to_id, id_to_tag, sess, model = n_service.create_instance()
            Singleton.__singleton = word_to_id, tag_to_id, id_to_tag, sess, model
        return Singleton.__singleton


@app.route("/")
def hello():
    import time
    s = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    return "Hello World!\n%s"%(s)


@app.route('/ner')
def ner():
    key = 'text'
    val = request.args.get(key)
    word_to_id, tag_to_id, id_to_tag, sess, model = Singleton.get_instance()

    batch = n_service.get_batch_manager(id_to_tag, tag_to_id, val, word_to_id)

    with sess.as_default():
        ner_results = model.predict2(sess, batch)

    labels = []
    datas = []
    for i, ret in enumerate(ner_results):
        strs = ret.split(" ")
        labels.append(strs[1])
        datas.append(strs[0])

    results = []
    for i in range(len(datas)):
        if labels[i] == "O":
            results.append(datas[i])
        else:
            results.append(datas[i]+labels[i])

    ret2 = " ".join(results)
    ret = "key:%s <br> <h4>value:%s</h4> "%(val, ret2)
    return ret

@app.route('/ner_text')
def ner_text():
    key = 'text'
    val = request.args.get(key)
    print "val-repr", repr(val)
    type = request.args.get('type')
    word_to_id, tag_to_id, id_to_tag, sess, model = Singleton.get_instance()

    batch = n_service.get_batch_manager(id_to_tag, tag_to_id, val, word_to_id)

    with sess.as_default():
        ner_results = model.predict2(sess, batch)

    datas = []

    val = val.encode('utf-8')
   # print "val-len", len(val)
    idx = 0
    is_flag = False
    item = {}
    for i, ret in enumerate(ner_results):
        strs = ret.split(" ")
        label = strs[1]
        data = strs[0].encode('utf-8') #中文3个，英文1个
        # print "label", label
        if label.startswith("B"):
            item = Item()
           # print "label[2:]",label[2:]
            #print "type",dict[label[2:]]
            item.type = dict[label[2:]]
            item.start = idx
            is_flag = True
            idx += len(data)
        elif label.startswith("I"):
            idx += len(data)

        elif is_flag and label.startswith("O"):# 遇到O结束了
            #idx += len(data)
            item.end = idx
            is_flag = False
            text = val[item.start:idx]
            item.text = text
            #print "text-mid", text
            # print "start", item.start, "end", idx, "text", text
            datas.append(item.to_string())
            idx += len(data)
        else:
            idx += len(data)

    if is_flag:
        item.end = idx
        text = val[item.start:idx]
        # print "start2", item.start, "end", idx, "text", text
        item.text = text
        #print "text-end",text
        # print json.dumps(item)
        datas.append(item.to_string())

    # d = {}
    # d["dict"] = datas
    # ret = json.dumps(datas)

    ret = ",".join(datas)
    ret = "{\"docs\":[%s]}"%(ret)
    print "ret", ret
    return ret

@app.route('/ner_text_prob')
def ner_text_prob():
    key = 'text'
    val = request.args.get(key)
    print "val-repr", repr(val)

    word_to_id, tag_to_id, id_to_tag, sess, model = Singleton.get_instance()

    batch = n_service.get_batch_manager(id_to_tag, tag_to_id, val, word_to_id)

    with sess.as_default():
        ner_results = model.predict_probility(sess, batch)

    datas = []

    val = val.encode('utf-8')
   # print "val-len", len(val)
    idx = 0
    is_flag = False
    item = {}
    for i, ret in enumerate(ner_results):
        strs = ret.split(" ")
        label = strs[1]
        data = strs[0].encode('utf-8') #中文3个，英文1个
        # print "label", label
        if label.startswith("B"):
            item = Item()
            item.type = dict[label[2:]]
            item.start = idx
            item.score = float(strs[2])
            is_flag = True
            idx += len(data)
        elif label.startswith("I"):
            idx += len(data)
            item.score *= float(strs[2])

        elif is_flag and label.startswith("O"):# 遇到O结束了
            #idx += len(data)
            item.end = idx
            is_flag = False
            text = val[item.start:idx]
            item.text = text
            item.score = np.power(item.score, float(1.0/len(text.decode('utf-8'))))
            # print "text-mid-size", len(text.decode('utf-8'))
            # print "start", item.start, "end", idx, "text", text
            datas.append(item.to_string())
            idx += len(data)
        else:
            idx += len(data)

    if is_flag: # 每一个score都计算了的
        item.end = idx
        text = val[item.start:idx]
        # print "start2", item.start, "end", idx, "text", text
        item.text = text
        item.score = np.power(item.score, float(1.0/len(text.decode('utf-8'))))

        datas.append(item.to_string())

    # d = {}
    # d["dict"] = datas
    # ret = json.dumps(datas)

    ret = ",".join(datas)
    ret = "{\"docs\":[%s]}"%(ret)
    print "ret", ret
    return ret


if __name__ == "__main__":
   
    app.run(host='XXXX', port=5005)

