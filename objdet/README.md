## introduction

one stage flow of object detection:

picture -> backbone --> feature-extraction -> bbox and kind 

1. training: bbox and kind -> loss -> backward propagation
2. evaluation: bbox and kind -> nms

|name|backbone|feature-extraction|
|-|-|-|
|YOLO-v1|inception-like|full-connection|
|SSD|vgg-like|FPN-like|

## experiment



## Reference

1. https://github.com/amdegroot/ssd.pytorch
2. https://github.com/xiongzihua/pytorch-YOLO-v1
