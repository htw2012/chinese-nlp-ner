#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2017-08-17
"""

import socket
import sys
sys.path.append('./gen-py')
from ner import NerService
from ner.ttypes import NerResponse, NerRequest, NerException,Item

import ner_service as n_service
import json
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
dict={"LOC":"Location","PER":"Person","ORG":"Organization"}
#import logging
#name = "ner_record"
#logger = logging.getLogger(name)
#logger.setLevel(logging.DEBUG)
#fh = logging.FileHandler(os.path.join("./logs", name + ".log"))
#fh.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.INFO)
#formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#ch.setFormatter(formatter)
#fh.setFormatter(formatter)
#logger.addHandler(ch)
#logger.addHandler(fh)

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

class NerServiceHandler:

     def process(self,req):
         res = NerResponse()

         val = req.query
         #print "get query", val
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
             data = strs[0].encode('utf-8')  # 中文3个，英文1个
             # print "label", label
             if label.startswith("B"):
                 item = Item()
                 item.type = dict[label[2:]]
                 item.start_idx = idx
                 is_flag = True
                 idx += len(data)
             elif label.startswith("I"):
                 idx += len(data)

             elif is_flag and label.startswith("O"):  # 遇到O结束了
                 # idx += len(data)
                 item.end_idx = idx
                 is_flag = False
                 text = val[item.start_idx:idx]
                 item.text = text.encode('utf-8')

                 datas.append(item)
                 idx += len(data)
             else:
                 idx += len(data)

         if is_flag:
             item.end_idx = idx
             text = val[item.start_idx:idx]
             item.text = text.encode('utf-8')
             datas.append(item)

         res.docs = datas
         res.status = 200
	 model.logger.info("req:{},res:{}".format(req.query,datas))
         return res

#创建服务端
handler = NerServiceHandler()
processor = NerService.Processor(handler)
#监听端口
#transport = TSocket.TServerSocket("59.110.213.232", 8205)
transport = TSocket.TServerSocket("172.17.54.175", 8206)
#选择传输层
tfactory = TTransport.TBufferedTransportFactory()
#选择传输协议
pfactory = TBinaryProtocol.TBinaryProtocolFactory()
#创建服务端
# server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
print "Starting thrift TThreadPoolServer in python..."
server.serve()
print "done!"
