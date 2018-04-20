import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import heatmap
import copy
import math 
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
        X[pt[1]-m:pt[1]+m,pt[2]-m:pt[2]+m] = color
    return X

def embed_fixations_gif(image_file, all_fixations,fixer,network,m = 2,color=(255,255,0)):
    X = cv2.imread(image_file)
    orig_shape = np.copy(X.shape)
    npad = ((0,513-X.shape[0]),(0,513-X.shape[1]),(0,0))
    #print(X.shape)
    X=np.pad(X, pad_width = npad , mode ='constant', constant_values = 0)
    # pad to get 513?
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,orig_shape[1])
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2


    # now we need to travel through all layers, add them to a resized map and rescale.
    duration_map = {'DEEPLAB_V2':0.3,'DEEPLAB_V1':0.4,'FCN':0.5}
    duration = duration_map[fixer.caffe_version]
    with imageio.get_writer('temp.gif', mode='I',duration =duration) as writer:
        key = fixer.top_layers[0]
        cur_fixations = {key:all_fixations[key]}
        reached_data = False
        done_list = []
        while (not reached_data):
            layer_names = cur_fixations.keys()
            if 'data' in layer_names:
                del cur_fixations['data']
            if len(cur_fixations) ==0:
                print('its over!')
                reached_data = True
                break
            if len(cur_fixations.keys())==1:
                name = cur_fixations.keys()[0]
                net_params = fixer.net[name]
                # save the pixelation
                size = network.blobs[name].data.shape[2]
                cur = cv2.resize(X,(size,size))
                for pt in cur_fixations[name]:
                    #print(pt)
                    cur[pt[1]-m:pt[1]+m,pt[2]-m:pt[2]+m] = color
                cur = cv2.resize(cur,(513,513))
                cur = cur[:orig_shape[0],:orig_shape[1],:]
                cv2.putText(cur,name, bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
                cv2.imwrite('xyz.png',cur)
                image = imageio.imread('xyz.png')
                writer.append_data(image)
                
                layer_below_name = net_params['bottom']
                cur_fixations = {name: all_fixations[name] for name in layer_below_name}
                done_list.append(name)
            else:
                layers_below = fixer._get_layers_below(cur_fixations.keys())
                to_remove = []
                print('at the start' , cur_fixations.keys())
                for name in cur_fixations.keys():
                    print(name,'going through')
                    net_params = fixer.net[name]
                    layer_top = net_params['name']

                    allow_flow = True
                    other_layers = [key for key in cur_fixations.keys() if key!= name]
                    if layer_top in layers_below:
                        print('In the root for some tree')
                        allow_flow = False
                        #break
                    if allow_flow and name not in done_list: # fish
                        print('Not the root for any tree!')
                        size = network.blobs[name].data.shape[2]
                        cur = cv2.resize(X,(size,size))
                        for pt in cur_fixations[name]:
                            #print(pt)
                            cur[pt[1]-m:pt[1]+m,pt[2]-m:pt[2]+m] = color
                        cur = cv2.resize(cur,(513,513))
                        cur = cur[:orig_shape[0],:orig_shape[1],:]
                        cv2.putText(cur,name, bottomLeftCornerOfText, font,fontScale,fontColor,lineType)
                        cv2.imwrite('xyz.png',cur)
                        image = imageio.imread('xyz.png')
                        writer.append_data(image)
                        
                        layer_below_name = net_params['bottom']
                        layer_below_name = [namez for namez in layer_below_name if namez in all_fixations.keys() and namez not in cur_fixations.keys()] # fish
                        new_fixations = {namez: all_fixations[namez] for namez in layer_below_name}
                        cur_fixations.update(new_fixations)
                        to_remove.append(name)
                        done_list.append(name)
                print('This is getting over',to_remove)
                to_remove = set(to_remove)
                for cur in to_remove:
                    #print(cur[0],len(cur[1]))
                    #print([[wur[0],len(wur[1])] for wur in cur_fixations])
                    del cur_fixations[cur]
                    print('removed',cur)
    
    
    return image
    
def get_heatmap(image_file, image_fixations):
    X = cv2.imread(image_file)
    orig_shape = np.copy(X.shape)
    npad = ((0,513-X.shape[0]),(0,513-X.shape[1]),(0,0))
    #print(X.shape)
    img=np.pad(X, pad_width = npad , mode ='constant', constant_values = 0)
    
    diag = math.sqrt(img.shape[0]**2 + img.shape[1]**2)*0.02
    fin_out = [[x[1],x[2]] for x in image_fixations]
    values = np.asarray(fin_out)
    neighbors = np.zeros((values.shape[0]))
    selPoints = np.empty((1,2))
    for i in range(values.shape[0]):
        diff = np.sqrt(np.sum(np.square(values-values[i]),axis=1))
        neighbors[i] = np.sum(diff<diag)
    for i in range(values.shape[0]):
        if neighbors[i]>0.05*values.shape[0]:
            selPoints = np.append(selPoints,values[i:i+1,:],axis=0)
    selPoints = selPoints[1:,:]
    selPoints[:,[0,1]] = selPoints[:,[1,0]]
    selPoints = selPoints.astype(int)

    hm = heatmap.Heatmap()
    ar = ((0, 0), (img.shape[1],img.shape[0]))
    si = (img.shape[1],img.shape[0])
    ds = int(75*((img.shape[0]+img.shape[1])/875.0))
    selPoints[:,1] = img.shape[0]-selPoints[:,1]
    heatMap = hm.heatmap(selPoints,area= ar,size = si,dotsize = ds,opacity=500)
    #To overlay on image
    heatMap = np.asarray(heatMap)
    indMap = heatMap[:,:,3]==0
    heatMap = heatMap[:,:,0:3]
    sup = cv2.addWeighted(img,0.4,heatMap,0.6,0)
    sup[indMap] = img[indMap]
    sup = sup[:orig_shape[0],:orig_shape[1],::-1]
    
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.add_axes(ax)
    plt.axis('off')
    plt.imshow(sup)
    plt.savefig('temp_heatmap.png',bbox_inches='tight', pad_inches=0)