# PDFNet-Pointwise-Dense-Flow-Network-for-Urban-Scene-Segmentation
# Abstract:
In recent years, using a deep convolutional neural network (CNN) as a feature encoder (or backbone) is the most commonly observed architectural pattern in several computer vision methods, and semantic segmentation is no exception. The two major drawbacks of this architectural pattern are: (i) the networks often fail to capture small classes such as wall, fence, pole, traffic light, traffic sign, and bicycle, which are crucial for autonomous vehicles to make accurate decisions. (ii) due to the arbitrarily increasing depth, the networks require massive labeled data and additional regularization techniques to converge and to prevent the risk of over-fitting, respectively. While regularization techniques come at minimal cost, the collection of labeled data is an expensive and laborious process. In this work, we address these two drawbacks by proposing a novel lightweight architecture named point-wise dense flow network (PDFNet). In PDFNet, we employ dense, residual, and multiple shortcut connections to allow a smooth gradient flow to all parts of the network. The extensive experiments on Cityscapes and CamVid benchmarks demonstrate that our method significantly outperforms baselines in capturing small classes and in few-data regimes. Moreover, our method achieves considerable performance in classifying out-of-the training distribution samples, evaluated on Cityscapes to KITTI dataset.
