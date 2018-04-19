# Seg-Unravel

This code contains implementation for **Segmentation-unravel**, something I attempted in my dissertation thesis, **"Per-Pixel Feedback for improving Semantic Segmentation"** [Arxiv Link](https://arxiv.org/abs/1712.02861). 

**Segmentation-Unravel** is an extension of [CNN-Fixations](https://arxiv.org/abs/1708.06670) for networks performing semantic segmentation. The motive for this work was 
* to see if global context played a role in segmentation, and 

* identifying if a few parts of object played a important role in segmentation. (For example a dog's head for dog's segmentation.)

**Segmentation-Unravel** backtracks the highest contributors of the segmentation output from the layer below. This unravelling process is continued till the image level to find 'fixation'-like points which might have been crucial for the segmentation.

This repository contains code for **Seg-Unravel** on the following Segmentation Networks (in caffe):

1) **fcn-alexnet** from "Fully Convolutional Networks for Semantic Segmentation". ([Arxiv link.](https://arxiv.org/abs/1411.4038))

2) **DeepLab VGG16 Large-FOV** from "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs". ([Arxiv Link](https://arxiv.org/pdf/1412.7062.pdf))

3) ** DeepLab Resnet 101 Multi-Scale** from "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs". ([Arxiv Link](https://arxiv.org/abs/1606.00915))


# Examples

Some images, Gifs, ==> image, seg-unravel process and then heatmaps.

# Instructions

First, run the `setup.sh` file, to 
* download the caffemodel and prototxt files for the segmentation models 
* download the caffe versions for each of these networks
* adapt the prototxt files for this project.

```
source setup.sh
```
** Kindly install caffe according to your setup(with/without CUDA, for instance). As there are way too many variables while installing caffe, it has not been automated.**

Now, we can run the `demo.py` file with various options to run **Seg-Unravel** on any selected network:
```
python demo.py 
```

This will save `temp.npy` with fixations at various layers, `temp_<class-name>.png` as image-level fixation map for the detected classes, and `temp.gif` as the fixation backtrack through layers animation for biggest segmented class(except background) in the input image. 

**Note** This code is a post-apocalypse recovery of code written during the project. It is not the cleanest code, nor the most lucid. However, it gets the job done, without errors. 

**Note** In context of segmentation, these 'fixations' might not be very useful. However, if anyone wants to use it, they may! 

# Acknowledgement

I would like to thank:

1) My Dissertation Guides, [Mopuri Konda Reddy](https://sites.google.com/site/kreddymopuri/) and [Ravi Kiran Sarvadevabhatla](https://ravika.github.io/).

2) [Utsav Garg](https://utsavgarg.github.io/) for the CNN-fixation code.

3) My Advisor for this project [Professor R. Venkatesh Babu](http://www.serc.iisc.ernet.in/~venky/).

4) Members of [Video Analytics Lab](http://val.serc.iisc.ernet.in/valweb/index.html) for general advice.