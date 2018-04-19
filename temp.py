execfile('seg_fix.py') 
fix = seg_fix(protofile,caffe_version)
top_fix = fix.get_top_fixations(x)
fix.set_top_layer()
fixations_top = [top_fix[0]]
output = fix.get_fixations_at_all_layers(fixations_top,network)

