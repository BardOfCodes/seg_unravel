import global_variables as g
caffe_name= 'FCN'
g.set_caffe_path(caffe_name)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np

# execfile('trick.py')
# execfile('seg_fix.py')

def get_net_structure(protofile,caffe_version):
    f = open(protofile,'r')
    solver = caffe_pb2.NetParameter()
    net_params =text_format.Merge(str(f.read()), solver)
    net_python = {}
    if caffe_version in ['BLVC', 'DEEPLAB_V2']:
        layer_obj = net_params.layer
    elif caffe_version == 'DEEPLAB_V1':
        layer_obj = net_params.layers
    num_layers = layer_obj.__len__()
    print(num_layers)
    for i in range(num_layers):
        cur_layer = layer_obj.pop()
        layer_params = get_params(cur_layer)
        #print(layer_params['name'])
        net_python[layer_params['name']]=layer_params
    return net_python


def get_params(obj):
    obj_dict = {}
    obj_list = obj.ListFields()
    for ob in obj_list:
        #ob = get_real_ob(ob)
        try:
            x=ob[1].ListFields()
            value = get_params(ob[1])
        except:
            value = ob[1]
        #if not ob[0].name in obj_dict.keys():
        #    #print(obj_dict.keys(), ob[0].name)
        obj_dict[ob[0].name]=value
        #else:
        #    print('here')
        #    if type(obj_dict[ob[0].name]) == list:
        #        obj_dict[ob[0].name].append(value)
        #    else:
        #        obj_dict[ob[0].name] = [obj_dict[ob[0].name],value]
    return obj_dict

def get_real_ob(ob):
    ob_new = []
    ob_new.append(ob[0])
    try:
        if not isinstance(ob[1], basestring):
            if len(ob[1])==1:
                ob_new.append(ob[1])
            else:
                ob_new.append(ob[1])
        else:
            ob_new.append(ob[1])
    except:
        ob_new.append(ob[1])
    return ob_new





class seg_fix:
    
    #initializer
    def __init__(self,protofile,caffe_version):
        # the net_params store all the important paramters from the a given
        # prototxt file
        self.net = get_net_structure(protofile,caffe_version)
        self.fixation_dict = {}


    def set_top_layer(self):
        net_layers = self.net.keys()
        net_tops = [self.net[i]['top'] for i in net_layers]
        net_tops = set([j for i in net_tops for j in i])
        net_bottoms = [self.net[i]['bottom'] for i in net_layers]
        net_bottoms = set([j for i in net_bottoms for j in i])
        self.top_layers = [key for key in net_layers if not key in net_bottoms]
        self.top_layer = top_layer
        

