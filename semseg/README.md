# Semantic Segmentation

## hybrid models
Based on vgg16 featuers,

1. SegNet

$Conv2D(\cat_i UpSampling(UnPool2D(feature_i)))$

2. PSPNet

$Conv2D(\cat_i UpSampling(MaxPool2D(feature_i,stride=1)))$

## Reference
1. https://github.com/bat67/pytorch-FCN-easiest-demo
2. https://github.com/pochih/FCN-pytorch
3. https://github.com/bodokaiser/piwise
4. https://github.com/jaxony/unet-pytorch/
