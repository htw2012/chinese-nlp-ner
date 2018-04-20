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

# myClassJson = json.dumps(myClassDict)

app = Flask(__name__)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

class Item(object):

    # 定义静态变量实例
    def __init__(self):
        self.start = -1
        self.end = -1
        self.text = ""
        self.type = ""

    def to_string(self):
        dict = {}
        dict["start"] = self.start
        dict["end"] = self.end
        dict["text"] = self.text
        dict["type"] = self.type
        #
        # line = ""
        # for key, value in dict.items():
        #     line += "\"%s\":\"%s\"" % (key, value)
        # return line
        str = json.dumps(dict, False, False, indent=4).encode("utf8")
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

    type = request.args.get('type')
    word_to_id, tag_to_id, id_to_tag, sess, model = Singleton.get_instance()

    batch = n_service.get_batch_manager(id_to_tag, tag_to_id, val, word_to_id)

    with sess.as_default():
        ner_results = model.predict2(sess, batch)

    datas = []

    idx = 0
    is_flag = False
    item = {}
    for i, ret in enumerate(ner_results):
        strs = ret.split(" ")
        label = strs[1]
        data = strs[0]
        # print "label", label
        if label.startswith("B"):
            item = Item()
            item.type = label[2:]
            item.start = idx
            is_flag = True
            idx += len(data)
        elif label.startswith("I"):
            idx += len(data)

        elif is_flag and label.startswith("O"):# 遇到O结束了
            idx += len(data)
            item.end = idx
            is_flag = False
            text = val[item.start:idx-1]
            item.text = text
            # print "start", item.start, "end", idx, "text", text
            datas.append(item.to_string())
        else:
            idx += len(data)

    if is_flag:
        item.end = idx
        text = val[item.start:idx]
        # print "start2", item.start, "end", idx, "text", text
        item.text = text
        # print json.dumps(item)
        datas.append(item.to_string())

    # d = {}
    # d["dict"] = datas
    # ret = json.dumps(datas)

    ret = ",".join(datas)
    ret = "{\"docs\":[%s]}"%(ret)
    print "ret", ret
    return ret

if __name__ == "__main__":
    # app.run(host='172.18.8.181', port=5001)
    app.run(host='localhost', port=5001)

