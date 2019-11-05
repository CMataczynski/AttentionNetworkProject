# Residual Attention Network for Image Classification
https://zpascal.net/cvpr2017/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf
## Attention-56
* 1  x Convolution: 7x7 window, 64 filters, stride 2
* 1 x Max Pooling: 3x3 window, stride 2
* 1 x Residual Unit: 64 filters, 1x1 window -> 64 filters, 3x3 window -> 256 filters, 1x1 window
* 1 x Attention Module: Mixed Attention
* 1 x Residual Unit: 128 filters, 1x1 window -> 128 filters, 3x3 window -> 512 filters, 1x1 window
* 1 x Attention Module: Mixed Attention
* 1 x Residual Unit: 256 filters, 1x1 window -> 256 filters, 3x3 window -> 1024 filters, 1x1 window
* 1 x Attention module: Mixed Attention
* 3 x Residual Unit: 512 filters, 1x1 window -> 512 filters, 3x3 window -> 2048 filters, 1x1 window
* 1 x Average Pooling: 7x7 window, stride 1
* Fully connected + softmax: 1000 neurons


## Attention-92
* 1  x Convolution: 7x7 window, 64 filters, stride 2
* 1 x Max Pooling: 3x3 window, stride 2
* 1 x Residual Unit: 64 filters, 1x1 window -> 64 filters, 3x3 window -> 256 filters, 1x1 window
* 1 x Attention Module: Mixed Attention
* 1 x Residual Unit: 128 filters, 1x1 window -> 128 filters, 3x3 window -> 512 filters, 1x1 window
* 2 x Attention Module: Mixed Attention
* 1 x Residual Unit: 256 filters, 1x1 window -> 256 filters, 3x3 window -> 1024 filters, 1x1 window
* 3 x Attention module: Mixed Attention
* 3 x Residual Unit: 512 filters, 1x1 window -> 512 filters, 3x3 window -> 2048 filters, 1x1 window
* 1 x Average Pooling: 7x7 window, stride 1
* Fully connected + softmax: 1000 neurons


## Attention Module:
We make the size of the smallest output map in each mask
branch 7×7 to be consistent with the smallest trunk output
map size. Thus 3,2,1 max-pooling layers are used in mask
branch with input size 56×56, 28×28, 14×14 respectively.

For structure see paper


## Training:
Nesterov SGD

mini-batch: 64

weight decay: 0.0001

momentum: 0.9

learning rate: 0.1 -64k iters-> 0.01 -96k iters-> 0.001

Learning time: 160k iters


# CBAM: Convolutional Block Attention Module
https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
## Used Architectures:
* ResNet18
* ResNet34
* ResNet50
* ResNet101
* WideResNet18 (widen = 1.5/2)
* ResNetXt50 (32x4d)
* ResNetXt101 (32x4d)
With addition of CBAM module, sequentially, channel attention first.
Fig 3 in paper for implementation details

## Training
Standard, with dropping learning rate every 30 epochs

# Attention gated networks: Learning to leverage salient regions in medical images
https://www.sciencedirect.com/science/article/pii/S1361841518306133
## Model
* 3 x Convolutional: 3x3 window, 8 filters, ReLu
* 1 x MaxPooling: /2
* 3 x Convolutional: 3x3 window, 16 filters, ReLu
* 1 x MaxPooling: /2
* 3 x Convolutional: 3x3 window, 32 filters, ReLu (1)
* 1 x MaxPooling: /2
* 2 x Convolutional: 3x3 window, 64 filters, ReLu (2)
* 1 x MaxPooling: /2
* 2 x Conbvolutional : 3x3 window, 64 filters, ReLu (3)
* (1) -> 1x64 filters
* (2) -> 1x64 filters
* (3) -> 1x64 filters
* Aggregation (1), (2), (3)
* Prediction
