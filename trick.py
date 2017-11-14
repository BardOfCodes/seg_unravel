import sys
sys.path.insert(0,'caffe/deeplab-public-ver2/python')
import caffe

import caffe
from caffe.proto import  caffe_pb2
from google.protobuf import text_format
import numpy as np
caffe_version = 'BLVC'
protofile = 'prototxt/resnet_msc.prototxt'
f = open(protofile,'r')
solver = caffe_pb2.NetParameter()
net_params =text_format.Merge(str(f.read()), solver)

