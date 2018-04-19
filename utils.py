import cv2
import numpy as np
import imageio

def get_blob(image_file):
    X = cv2.imread(image_file)
    # Flip for cv2 read
    X = np.copy(X[:,:,::-1])
    X = X.astype(float)
    X = X.swapaxes(0,2).swapaxes(1,2)
    X[0] = X[0] - 104.008
    X[1] = X[1] - 116.669
    X[2] = X[2] - 122.675
    npad = ((0,0),(0,513-X.shape[1]),(0,513-X.shape[2]))
    X=np.pad(X, pad_width = npad , mode ='constant', constant_values = 0)
    X = X.astype(float)
    X = np.expand_dims(X, axis=0)
    X = np.array(X,dtype='f')
    return X

def embed_fixations(image_file, image_fixations,m = 5,color=(255,255,0)):
    X = cv2.imread(image_file)
    for pt in image_fixations:
        #print(pt)
        X[pt[1]-2:pt[1]+2,pt[2]-2:pt[2]+2] = color
    return X

def embed_fixations_gif(image_file, all_fixations):
    X = cv2.imread(image_file)
    # now we need to travel through all layers, add them to a resized map and rescale.
    
    
    return image
    
