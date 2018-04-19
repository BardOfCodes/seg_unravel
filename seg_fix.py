import global_variables as g
#g.set_caffe_path(caffe_name)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
from google.protobuf.pyext._message import RepeatedScalarContainer as RSC
from google.protobuf.pyext._message import RepeatedCompositeContainer as RCC
import copy

class seg_fix:
    
    #initializer
    def __init__(self,protofile,caffe_version,shift_type= 'full_shift'):
        # the net_params store all the important paramters from the a given
        # prototxt file
        self.net = self._get_net_structure(protofile,caffe_version)
        for layer in self.net.keys():
            if self.net[layer]['type'] in ['ReLU','Scale','BatchNorm']: 
                del self.net[layer]
        self.net[u'data']= {'bottom':['data'],'top':['conv1']}
        # remove all the ReLU, BN,SCALE.
        self.fixation_dict = {}
        self.DILATION = 'dilation'
        self.KERNEL = 'kernel_size'
        self.caffe_version = caffe_version
        self.shift_type = shift_type
        self.set_top_layer()

    def _get_net_structure(self,protofile,caffe_version):
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
            layer_params = self._get_params(cur_layer)
            net_python[layer_params['name']]=layer_params
        return net_python

    def _get_params(self,obj):
        obj_dict = {}
        obj_list = obj.ListFields()
        for ob in obj_list:
            try:
                x=ob[1].ListFields()
                value = self._get_params(ob[1])
            except:
                if isinstance(ob[1],RSC) or isinstance(ob[1],RCC):
                    #print('RSC or RCC',ob[0].name,ob[1])
                    value = [x for x in ob[1]]
                else:
                    #print(ob[0].name,ob[1])
                    value = ob[1]

            obj_dict[ob[0].name]=value
        return obj_dict
    
    def set_top_layer(self):
        net_layers = self.net.keys()
        net_tops = [self.net[i]['top'] for i in net_layers]
        net_tops = set([j for i in net_tops for j in i])
        net_bottoms = [self.net[i]['bottom'] for i in net_layers]
        net_bottoms = set([j for i in net_bottoms for j in i])
        self.top_layers = [key for key in net_tops if not key in net_bottoms]
        #self.top_layer = top_layer
    def _get_layers_below(self, layer_names):
        layers_below_all = []
        while layer_names != []:
            #print(layer_names)
            cur_layers_below = []
            for layer in layer_names:
                #print(self.net[layer]['bottom'])
                cur_layers_below.append(self.net[layer]['bottom'])
            cur_layers_below = list(set([j for i in cur_layers_below for j in i]))
            layer_names =  [name for name in cur_layers_below if not name in layers_below_all]
            layers_below_all += cur_layers_below
        layers_below_all = set(layers_below_all)
        #print(layers_below_all)
        return layers_below_all

    def _checker_func(self,params,names,defaults):
        val_list = []
        print(params.keys())
        print(names)
        for i,name in enumerate(names):
            if name in params.keys():
                if isinstance(params[name],list):
                    val_list.append(params[name][0])
                else:
                    val_list.append(params[name])
            else:
                val_list.append(defaults[i])
        print(val_list)
        return val_list
    
    def get_top_fixations(self,output_blob):
        output_prediction = np.argmax(output_blob,axis=1)[0]
        classes_detected= np.unique(output_prediction)
        fixation_dict = {}
        for class_id in classes_detected:
            class_fix = np.where(output_prediction==class_id)
            count = np.arange(len(class_fix[0]))
            class_fix = list(set([(class_id,class_fix[0][i],class_fix[1][i]) for i in
                              count]))
            fixation_dict.update({class_id:class_fix})
        return fixation_dict

    def get_fixations_at_all_layers(self,fixations_top,network,save_all= False):
        # make the layer by layer iteration
        fixations_map = {'Convolution':self.conv_fixations,
                         4:self.conv_fixations, 
                         'Interp':self.interp_fixations,
                         41:self.interp_fixations,
                         'Pooling':self.pool_fixations,
                         17:self.pool_fixations,
                         'Eltwise':self.eltwise_fixations,
                         'Crop':self.crop_fixations}
        cur_fixations = {self.top_layers[0]:fixations_top}
        #print(cur_fixations[0][0])
        reached_data = False
        image_level_fixations = []
        all_fixations = {}
        while (not reached_data):
            layer_names = cur_fixations.keys()
            #print(layer_names)
            if 'data' in layer_names:
                image_level_fixations.append(cur_fixations['data'])
                del cur_fixations['data']
            if len(cur_fixations) ==0:
                print('its over!')
                reached_data = True
                break
            if save_all:
                all_fixations.update(cur_fixations)
            if len(cur_fixations.keys())==1:
                print('len is 1')
                name = cur_fixations.keys()[0]
                net_params = self.net[name]
                print(name,'going through')
                transfer_type = net_params['type']
                print('transfer_type',transfer_type)
                fixations_below = fixations_map[transfer_type](cur_fixations[name],net_params,network)
                #print(fixations_below[0][0])
                cur_fixations = fixations_below
                print('reached_data',reached_data)
            else:
                print('len is',len(cur_fixations.keys()))
                layers_below = self._get_layers_below(cur_fixations.keys())
                #layers_below = [j for i in layers_below for j in i]
                print('layers_below',len(layers_below))
                to_remove = []
                for name in cur_fixations.keys():
                    print(name,'going through')
                    cur = cur_fixations[name]
                    net_params = self.net[name]
                    print('net_params',net_params,net_params['type'])
                    layer_top = net_params['name']

                    allow_flow = True
                    other_layers = [key for key in cur_fixations.keys() if key!= name]
                    if layer_top in layers_below:
                        print('In the root for some tree')
                        allow_flow = False
                        #break
                    if allow_flow:
                        print('Not the root for any tree!')
                        transfer_type = net_params['type']
                        print('transfer_type',transfer_type)
                        fixations_below = fixations_map[transfer_type](cur,net_params,network)
                        all_keys = [key for key in cur_fixations.keys()]
                        print('all_keys',all_keys)
                        for fix_name,fix in fixations_below.iteritems():
                            print('got',fix_name,'with no of fixations',len(fix))
                            if fix_name in all_keys:
                                #all_fixations = set(cur[layers.index(fix[0])][1].list() +fix[1].list())
                                print(len(cur_fixations[fix_name]))
                                cur_fixations[fix_name] =list(set(cur_fixations[fix_name]+ fix))
                            else:
                                #to_remove.append(cur_fixations[layers.index(fix[0])])
                                cur_fixations.update({fix_name:fix})
                        to_remove.append(name)
                to_remove = set(to_remove)
                for cur in to_remove:
                    #print(cur[0],len(cur[1]))
                    #print([[wur[0],len(wur[1])] for wur in cur_fixations])
                    del cur_fixations[cur]
                    print('removed',cur)
        image_level_fixations = [j for i in image_level_fixations for j in i]
        return image_level_fixations, all_fixations

    def crop_fixations(self,cur_fixations,cur_params,network):
        # TO DO
        return None
    def deconv_fixations(self,cur_fixations,cur_params,network):
        # TO DO
        return None

    def interp_fixations(self,cur_fixations,cur_params,network):
        zoom_factor, shrink_factor = self._checker_func(cur_params['interp_param'],['zoom_factor','shrink_factor'],[1,1])
        #print(cur_fixations)
        y_vector = np.array([x[1] for x in cur_fixations])
        x_vector = np.array([x[2] for x in cur_fixations])
        y_vector = y_vector/zoom_factor*shrink_factor
        x_vector = x_vector/zoom_factor*shrink_factor
        fix_list = [(cur_fixations[x][0],y_vector[x],x_vector[x]) for x in range(y_vector.shape[0])]
        fix_list = list(set(fix_list))
        name = cur_params['bottom'][0]
        return  {name:fix_list}

    def eltwise_fixations(self,cur_fixations,cur_params,network):
        layers_below = cur_params['bottom']
        outputs = np.zeros((len(cur_fixations),len(layers_below)))
        print(layers_below,'branch_below')
        for i,fixation in enumerate(cur_fixations):
            for j,layer in enumerate(layers_below):
                #print(layer,network.blobs[layer].data.shape,fixation)
                outputs[i,j] = network.blobs[layer].data[0][fixation]
        max_outputs = np.argmax(outputs,1)
        fix_list = [[None] for i in layers_below]
        for i,fixation in enumerate(cur_fixations):
            fix_list[max_outputs[i]].append(fixation)
        global_fix_list = {}
        for i,layer in enumerate(layers_below):
            del fix_list[i][0]
            print(len(fix_list[i]))
            if len(fix_list[i])!=0:
                global_fix_list.update({layer:fix_list[i]})
        return global_fix_list

    # Similar Functions for Pool
    def pool_fixations(self,cur_fixations,cur_params,network,K=1):
        # Parameters : K top_k
        #              S Stride
        #              filter_size: kernel size
        #              pad = padding!!!
        
        pooling_params = cur_params['pooling_param']
        value_names = ['stride','pad',self.KERNEL]
        defaults = [1,0,2]
        S,filter_size,pad = self._checker_func(pooling_params,value_names,defaults)
        
        npad = ((0,0),(pad,pad),(pad,pad))
        blob_below = network.blobs[cur_params['bottom'][0]].data[0]
        
        blob_below_pad =np.pad(blob_below, pad_width = npad , mode ='constant', constant_values = 0)
        
        fixations_below_list = []
        for point in cur_fixations:
            # pool will just reduce the spatial dimensions of blob.
            z = point[0]
            y = point[1]
            x = point[2]
            #print(point)
            blob_below_cur = blob_below_pad[point[0]:point[0]+1,y*S:y*S + filter_size,x*S:x*S + filter_size]
            
            list_maxer = np.dstack(np.unravel_index(np.argsort(blob_below_cur.ravel()), blob_below_cur.shape))
            # add k should be less than filter_size*filter_size
            top_points = list_maxer[0][-K:]
            top_points = [(z,min(max(0,point[1]+ y*S-pad),blob_below.shape[1]-1),min(max(0,point[2]+x*S-pad),blob_below.shape[2]-1) ) for point in top_points]
            fixations_below_list.extend(top_points)
            #print('output',top_points)
        fixations_below_list = list(set(fixations_below_list))
        name = cur_params['bottom'][0]
        return {name:fixations_below_list}

    def conv_fixations(self,cur_fixations, cur_params, network ,K=1):
        # fixation_above_list is the fixations in the list above
        # Parameters : K ==> top_k
        #              S ==> Stride
        #              pad ==> padding for the conv layer
        #              group ==> grouping
        #              hole/dilation ==> a-trous convolution
        # We can add more later
        convolution_params = cur_params['convolution_param']
        name_params = ['stride','pad','group',self.DILATION]
        defaults = [1,0,1,1]
        S,pad,group,hole =self._checker_func(convolution_params,name_params,defaults)
        #if self.caffe_version == 'DEEPLAB_V2':
        #    pad = int(pad[0])
        #    hole = int(hole[0])
        #    S = int(S[0])

        if self.shift_type=='full_shift':
            allow_shift = True
        elif self.shift_type == 'no_shift':
            allow_shift = False
        else:
            if hole == 1:
                allow_shift = True
            else:
                allow_shift = False
        fixations_below_list = []
        # make the pad before itself!
        npad = ((0,0),(pad,pad),(pad,pad))
        blob_below = network.blobs[cur_params['bottom'][0]].data[0]
        
        blob_below_pad =np.pad(blob_below, pad_width = npad , mode ='constant', constant_values = 0)
        
        # for groups.
        z_blob_above = network.blobs[cur_params['name']].data[0].shape[0]
        z_per_group = z_blob_above/group
        
        z_blob_below = blob_below.shape[0]
        z_per_group_below = z_blob_below/group
            
        for point in cur_fixations:
            #print(point)
            filter_res = point[0]
            filter_params = network.params[cur_params['name']][0].data[filter_res]
            
            # conv filter dim etc will be in filter dims
            filter_dims = filter_params.shape[0:3]
            
            z = point[0]
            x = point[2]
            y = point[1]
            group_num = int(z/z_per_group)
            
            blob_below_cur = np.copy(blob_below_pad[group_num*z_per_group_below:(group_num+1)*z_per_group_below,y*S:y*S+filter_dims[1],x*S:x*S+filter_dims[2]])
            
            if (hole!=1):
                blob_below_cur_hole = blob_below_pad[group_num*z_per_group_below:(group_num+1)*z_per_group_below,y*S:y*S+filter_dims[1]+(filter_dims[1]-1)*(hole-1),x*S:x*S+filter_dims[2]+(filter_dims[2]-1)*(hole-1)]
                # now replace each z-vector by the correct one
                for row in range(filter_dims[1]):
                    for col in range(filter_dims[2]):
                        # assuming symmetric x,y dim of filter
                        # for odd filter_size
                        try:
                            blob_below_cur[:,row,col] =blob_below_cur_hole[group_num*z_per_group_below:(group_num+1)*z_per_group_below,row*hole,col*hole] 
                        except IndexError:
                            print(str(point))
                            print(str(blob_below_cur_hole.shape))
                            print(str(blob_below_cur.shape))
            
            # Now calculating Output
            try:
                outputs = blob_below_cur*filter_params
            except ValueError:
                print('At Layer ' + str(layer_index))
                print(str(point))
            ########################
            feature=np.sum(np.sum(outputs,axis=2),axis=1)
            #feature.shape
            # Add function for K
            # we used functions for changing K in the thesis.
            #K = 1
            top_layers = np.argsort(feature.ravel())[-K:]
            top_points = []
            for layer in top_layers:
                if(allow_shift):
                    output = outputs[layer]
                    list_maxer = np.dstack(np.unravel_index(np.argsort(output.ravel()), output.shape))
                    top_point = list_maxer[0][-1]
                    top_points.append([layer,top_point[0],top_point[1]])
                else:
                    top_points.append([layer,int(filter_dims[1]/2),int(filter_dims[1]/2)])
            ########################
            list_points = []
            for point in top_points:
                pointer = [(point[0]+group_num*z_per_group_below,min(max(0,y*S+(point[1])*(hole)-pad),blob_below.shape[1]-1),min(max(0,x*S+(point[2])*(hole)-pad),blob_below.shape[2]-1))]
                list_points.extend(pointer)
            top_points = list_points
            fixations_below_list.extend(top_points)
            #print(top_points)
        fixations_below_list = list(set(fixations_below_list))
        name = cur_params['bottom'][0]
        return {name:fixations_below_list}
