# First Download all the requied files.
# FCN - Alexnet
wget https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn-alexnet/val.prototxt
mv val.prototxt prototxt/fcn_alexnet.prototxt
wget http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel
mv fcn-alexnet-pascal.caffemodel weights/fcn_alexnet.caffemodel
# Deeplab V1 VGG16 Large FOV
wget http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/test.prototxt
wget http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/train2_iter_8000.caffemodel
mv test.prototxt prototxt/dlvgglfov.prototxt
mv train2_iter_8000.caffemodel weights/dlvgglfov.caffemodel
# Deeplab V2 Multi-Scale-Resnet-101
wget http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip
unzip prototxt_and_model.zip
mv test.prototxt prototxt/resnet_msc.prototxt
mv train2_iter_20000.caffemodel weights/resnet_msc.prototxt
rm *.prototxt
rm *.zip
rm *.caffemodel

# Clone the deeplab repositories 
mkdir -p caffe
cd caffe
git clone https://github.com/BVLC/caffe.git
# Compile caffe by following the instructions at https://github.com/BVLC/caffe
git clone https://bitbucket.org/deeplab/deeplab-public.git
# Compile deeplab v1 by following the instrctions at https://bitbucket.org/deeplab/deeplab-public/
git clone https://bitbucket.org/aquariusjay/deeplab-public-ver2.git
# Compile deeplab v2 by following the instructions at https://bitbucket.org/aquariusjay/deeplab-public-ver2
cd ../
# Remeber to Complile PyCaffe as well!

# Modify the prototxt file to remove the preprocessing layers:
# direct input prtoto

export proto="input:\"data\"\ninput_dim:1\ninput_dim:3\ninput_dim:513\ninput_dim:513\n"
# FCN
sed -i '263,273d' prototxt/fcn_alexnet.prototxt
sed -i '1,11d' prototxt/fcn_alexnet.prototxt
sed -i "1i ${proto}" prototxt/fcn_alexnet.prototxt
# DlVggLfov
sed -i '414,425d' prototxt/dlvgglfov.prototxt
sed -i '11,31d' prototxt/dlvgglfov.prototxt
sed -i "11i ${proto}" prototxt/dlvgglfov.prototxt
sed -i -e 's/${EXP}/voc12/g' prototxt/dlvgglfov.prototxt 
sed -i -e 's/${NUM_LABELS}/21/g' prototxt/dlvgglfov.prototxt 
sed -i -e 's/${NET_ID}/dlvgglfov/g' prototxt/dlvgglfov.prototxt 
sed -i -e 's/dropout_ratio: 0.5/dropout_ratio: 0.0/g' prototxt/dlvgglfov.prototxt
# Resnet_Msc
sed -i '21825,21843d' prototxt/resnet_msc.prototxt
sed -i '3,26d' prototxt/resnet_msc.prototxt
sed -i "3i ${proto}" prototxt/resnet_msc.prototxt
sed -i -e 's/${EXP}/voc12/g' prototxt/resnet_msc.prototxt 
sed -i -e 's/${NUM_LABELS}/21/g' prototxt/resnet_msc.prototxt 
sed -i -e 's/${NET_ID}/resnet_msc/g' prototxt/resnet_msc.prototxt 
sed -i $'s/\r//' prototxt/resnet_msc.prototxt
