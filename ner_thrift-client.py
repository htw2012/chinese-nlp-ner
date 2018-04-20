#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2017-08-17
"""
import sys
import glob
import sys
sys.path.append('./gen-py')
from ner import NerService
from ner.ttypes import NerResponse, NerRequest, NerException,Item

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


def main():
    # Make socket
    transport = TSocket.TSocket('XXX', 8205)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = NerService.Client(protocol)


    # Connect!
    transport.open()
    req = NerRequest()
    req.query = "导航去知春路"

    try:
        res = client.process(req)

        st = res.status
        print "status", st
        docs = res.docs
        for doc in docs:
            print "start", doc.start_idx, "end", doc.end_idx, "text", doc.text, "type", doc.type

    except NerException as e:
        print('InvalidOperation: %r' % e)

    # Close!
    transport.close()

main()