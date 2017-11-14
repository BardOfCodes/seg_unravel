import sys
import os



# Path to the caffe files
DEEPLAB_V1_PATH = os.path.join('caffe','deeplab-public','python')
DEEPLAB_V2_PATH = os.path.join('caffe','deeplab-public-ver2','python')
FCN_PATH = os.path.join('caffe','caffe','python')

def set_caffe_path(caffe_name):
    if caffe_name == 'DEEPLAB_V1':
        sys.path.insert(0,DEEPLAB_V1_PATH)
    elif caffe_name == 'DEEPLAB_V2':
        sys.path.insert(0,DEEPLAB_V2_PATH)
    elif caffe_name == 'FCN':
        sys.path.insert(0,FCN_PATH)


