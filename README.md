# Large Margin In Softmax Cross-Entropy Loss

The Pytorch implementation for the BMVC2019 paper of "[Large Margin In Softmax Cross-Entropy Loss](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/BMVC2019.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2019bmvc,
  title={Large Margin In Softmax Cross-Entropy Loss},
  author={Takumi Kobayashi},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2019}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)

## Introduction

The proposed method works as a regularization for the standard softmax cross-entropy loss to promote the large-margin networks.
So, it is noteworthy that the large margin can be embedded into neural networks, such as CNNs, by simply adding the proposed regularization without touching other components; we can use the same training procedures, such as optimizer, learning rate and training schedule.
For the more detail, please refer to our [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/BMVC2019.pdf).

<img width=500 src="https://user-images.githubusercontent.com/53114307/64231100-9f9d3680-cf29-11e9-83b3-402c820d2cad.png">

Figure: Comparison of large-margin losses

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(>=1.0.0)](http://pytorch.org)

### Training
The softmax loss with the large-margin regularization can be simply incorporated by

```python
from models.modules.myloss import LargeMarginInSoftmaxLoss
criterion = LargeMarginInSoftmaxLoss(reg_lambda=0.3)
```

where `reg_lambda` indicates the regularization parameter.

For example, the 13-layer network is trained on Cifar10 by using the following command

```bash
CUDA_VISIBLE_DEVICES=0 python cifar_train.py  --dataset cifar10  --data ./datasets/ --arch layer13  --config-name layer13_largemargin  --out-dir ./result/cifar10/layer13/LargeMarginInSoftmax/
```

The VGG-16 mod network [1] on ImageNet is also trained by

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python imagenet_train.py  --dataset imagenet  --data ./datasets/imagenet12/images/  --arch vgg16bow_bn  --config-name imagenet_largemargin  --out-dir ./result/imagenet/vgg16bow_bn/LargeMarginInSoftmax/  --dist-url 'tcp://127.0.0.1:8080'  --dist-backend 'nccl'  --multiprocessing-distributed  --world-size 1  --rank 0
```

Note that the imagenet dataset must be downloaded at `./datasets/imagenet12/` before the training.

### Results
These performance results are not the same as those reported in the paper because the methods are implemented by MatConvNet in the paper and accordingly trained in a (slightly) different training procedure.

#### Cifar-10

| Network  | Loss | Top-1 Err. |
|---|---|---|
| 13-Layer|  SoftMax | 8.45 (+-0.27)|
| 13-Layer|  SoftMax with Large-Margin | 7.81 (+-0.20)|

#### Cifar-100

| Network  | Loss | Top-1 Err. |
|---|---|---|
| 13-Layer|  SoftMax | 29.42 (+-0.19)|
| 13-Layer|  SoftMax with Large-Margin | 27.61 (+-0.09)|

#### ImageNet

| Network  | Loss | Top-1 Err. |
|---|---|---|
| VGG-16 mod [1]|  SoftMax | 22.99 |
| VGG-16 mod [1]|  SoftMax with Large-Margin | 22.09 |
| VGG-16 [2]|  SoftMax | 25.04 |
| VGG-16 [2]|  SoftMax with Large-Margin | 24.08 |
| ResNet-50 [3]|  SoftMax | 23.45 |
| ResNet-50 [3]|  SoftMax with Large-Margin | 23.28 |
| ResNeXt-50 [4]|  SoftMax | 22.42 |
| ResNeXt-50 [4]|  SoftMax with Large-Margin | 22.27 |
| DenseNet-169 [5]|  SoftMax | 23.03 |
| DenseNet-169 [5]|  SoftMax with Large-Margin | 22.70 |

## References:

[1] T. Kobayashi. "Analyzing Filters Toward Efficient ConvNets." In CVPR, pages 5619-5628, 2018. [pdf](https://staff.aist.go.jp/takumi.kobayashi/publication/2018/CVPR2018.pdf)

[2] K. Simonyan and A. Zisserman. "Very Deep Convolutional Networks For Large-Scale Image Recognition." CoRR, abs/1409.1556, 2014.

[3] K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning For Image Recognition." In CVPR, pages 770–778, 2016.

[4] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. "Aggregated Residual Transformations For Deep Neural Networks." In CVPR, pages 5987–5995, 2017.

[5] G. Huang, Z. Liu, L. Maaten and K.Q. Weinberger. "Densely Connected Convolutional Networks." In CVPR, pages 2261-2269, 2017.