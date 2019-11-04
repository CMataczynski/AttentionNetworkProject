# Attention-56
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


# Attention-92
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


#Attention Module:
We make the size of the smallest output map in each mask
branch 7×7 to be consistent with the smallest trunk output
map size. Thus 3,2,1 max-pooling layers are used in mask
branch with input size 56×56, 28×28, 14×14 respectively.

For structure see paper


#Training:
Nesterov SGD
mini-batch: 64
weight decay: 0.0001
momentum: 0.9
learning rate: 0.1 -64k iters-> 0.01 -96k iters-> 0.001
Learning time: 160k iters
