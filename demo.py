'''
Script containing a demo run for seg-unravel
To use:
python demo.py --network <network> --shift_type <shift-type> --image <image-path>

<network> DEEPLAB_V1 or DEEPLAB_V2 or FCN
<shift-type> no_shift or partial_shift or full_shift
<image-path> path to image file.
'''
import argparse
from global_variables import *
import utils as u
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description='Seg-unravel on a given image.')
    parser.add_argument('--network',               type=str,
                        help='The segmentation network to be used.', required=True)
    parser.add_argument('--shift_type', type=str,
                        help='The shift type to be used. Read Thesis for more details.',default='full_shift')
    parser.add_argument('--image',               type=str,
                        help='Path to image file.', required=True)
    
    args = parser.parse_args()
    
    # set caffe before importing seg_fix
    caffe_version = args.network
    set_caffe_path(caffe_version)
    import seg_fix
    import caffe
    # get the caffe model
    prototxt = os.path.join('prototxt',file_map[caffe_version][0])
    wts = os.path.join('weights',file_map[caffe_version][1])
    
    
    net = caffe.Net(prototxt,wts, caffe.TEST)
    
    # For forward
    image_blob = u.get_blob(args.image)
    net.blobs['data'].data[...] = image_blob
    top_layer_output = net.forward()['fc1_interp']
    
    # Seg_fix in action
    fixer = seg_fix.seg_fix(prototxt,caffe_version)
    top_fixations = fixer.get_top_fixations(top_layer_output)
    
    # find class-id
    seg_map = np.argmax(top_layer_output,1)[0]
    class_ids = list(np.unique(seg_map))
    class_ids.remove(0) # not bg
    seg_space = [np.sum(seg_map==id) for id in class_ids]
    class_id = class_ids[np.argmax(seg_space)]
    
    # you can get fixations for each of the detected classes by sending each detected class-id in the following function
    image_fixations, all_fixations= fixer.get_fixations_at_all_layers(top_fixations[class_id],net)
    
    # save numpy with fixations 
    np.save('temp.npy',all_fixations)
    
    # Get the image_with Fixations
    img_with_fixations = u.embed_fixations(args.image,image_fixations)
    cv2.imsave('temp.png', img_with_fixations)
    
    # get the gif of fixation flow:
    #img_with_fixations = u.embed_fixations_gif(args.image, all_fixations)
    # this saves the gif as temp.gif

if __name__ == '__main__':
    main()