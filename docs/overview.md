## Papers：

> Image [Style Transfer](https://zhida.zhihu.com/search?content_id=108596526&content_type=Article&match_order=1&q=Style+Transfer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjYyMjcxNTcsInEiOiJTdHlsZSBUcmFuc2ZlciIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjEwODU5NjUyNiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.6C9R29anw0c90M9rIg7edFqIhFHa1TdUr0dy_tE9ZMw&zhida_source=entity) Using [Convolutional Neural Networks](https://zhida.zhihu.com/search?content_id=108596526&content_type=Article&match_order=1&q=Convolutional+Neural+Networks&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjYyMjcxNTcsInEiOiJDb252b2x1dGlvbmFsIE5ldXJhbCBOZXR3b3JrcyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjEwODU5NjUyNiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.cLA5HYGPP1Z78VLtjf-sPUb2qHvZgZb_wFcgcb85pPg&zhida_source=entity)  
> Perceptual Losses for Real-Time Style Transfer and Super-Resolution

开源项目地址：[https://github.com/GitHberChen/GAN/tree/master/src/style\_transfer\_perceptual\_loss](https://link.zhihu.com/?target=https%3A//github.com/GitHberChen/GAN/tree/master/src/style_transfer_perceptual_loss)

后续提供详细使用方法。

## 一、什么是图像风格迁移（Style Transfer）

所谓风格迁移，就是让一张图片具有其原本的内容 content，同时具有另一张图片（通常是艺术作品）的风格 style，如下图。

这个领域的研究与大家生活最接近的当属手机APP，Prisma了，想起多年前使用这款APP的最大感受就是：慢！烫！修改一张图具有某种风格要很长时间，并且十分耗电。而今天介绍的paper 是Li Fei-Fei 大弟子 Justin Johnson，于 2016ECCV 发表的一篇 Paper，将原先十分耗时的风格迁移计算提速了 1000x 倍，与此同时保持相同的效果。

## 二、如何使用CNN进行风格迁移？

一个经典的 style transfer 算法来自一篇 CVPR2016 的Paper，大家感受一下：

> 参考：Image Style Transfer Using Convolutional Neural Networks

看上去是不是很复杂？简单来说，这个算法流程是这样的：

1、使用一个pre-trained的[VGG](https://zhida.zhihu.com/search?content_id=108596526&content_type=Article&match_order=1&q=VGG&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjYyMjcxNTcsInEiOiJWR0ciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMDg1OTY1MjYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.640n3ZjlhDo9duATAFZuJK5pneBwE60ju54AtVVDb3o&zhida_source=entity)，将其看做一个丰富特征提取 filter 的集合；

2、使用一张图片 a 定义一个目标风格，p 为原始图片， x 为输出图片，而 x 初始化为 white noise 图片；

3、定义一个风格损失函数 Lstyle 、内容损失函数 Lcontent ，将 x 在 VGG 前向传播的特征图与 p 的特征图使用 Lcontent 对比，与 a 的特征图使用 Lstyle 对比；然后将差别反向传播给 x，注意 VGG 是固定的；

4、一次又一次的迭代后， white noise x 会逐渐产生我们想要的图片：具有 p 的内容，同时具有 a 的风格。

显然，既然要一次又一次前向反向传播迭代，那耗时当然不会短。那有没有办法更快地得到结果呢？

Perceptual Losses for Real-Time Style Transfer and Super-Resolution 这篇 paper 就以 1000x 的提速，解决了这个问题。思路也很简单，回避迭代的方法，使用司空见惯的 end-to-end training的方法，训练一个[DCNN](https://zhida.zhihu.com/search?content_id=108596526&content_type=Article&match_order=1&q=DCNN&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjYyMjcxNTcsInEiOiJEQ05OIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTA4NTk2NTI2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.pLxxjFjkVDCnPewFo5gF2knHXkZ-sI5GsWs55711NIY&zhida_source=entity)模型，而这个模型输入原始图片后可以得到我们想要的具有某种特殊风格的图片，整体算法流程如下。而虽然说起来简单，但最难的地方在于如何定义loss，接下来将会介绍它是如何做到的。

速度提升对比：

要想对图像进行风格迁移，首先必须要做的是定义风格是什么？

风格是什么，作为一个对世界有着丰富而敏感的人，当然可以感受得到，可要想用言语精确描述达到可以量化的程度，我想是难以做到的。于是问题就来了，如果我们无法精确地量定义风格是什么，又该如何去教机器去了解什么是风格呢？

为了解决这个问题，学者们采用了一个十分取巧的方法，那就是，既然我们无法定义风格是什么，那么不妨定义一下：风格不是什么？

**那么风格不是什么呢？**

风格绝对不是内容，即同样一副美术作品的内容，是可以用不同风格来表达的，而具有同样风格的作品，可以具有完全不同的内容。更具体地说，风格是一种特征，这种特征具有**位置不敏感性。**

于是，我们可以借用训练好的 DCNN ，前向传播提取图片的特征图 F∈RC×H×W后，对于每个点的特征 Fh,w∈RC 求其 Gram 矩阵得到 Gh,w\=Fh,wFh,wT∈RC×C ，然后将每个点的 [Gram 矩阵](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/%25E6%25A0%25BC%25E6%258B%2589%25E5%25A7%2586%25E7%259F%25A9%25E9%2598%25B5)相加 G\=∑h,wGh,w\=∑h,wFh,wFh,wT∈RC×C 这个 Gram 矩阵最大的特点就是具有**位置不敏感性**， 所以，我们可以将这个 G 当做衡量一张图篇风格的量化描述，考虑到一个卷积神经网络中间有多层 特征图，对于每层特征图都可以得到 Gram 矩阵，所以我们可以使用 {G1,G2,...Gl} 来更为全面地描述一张图的风格。相应的损失函数则为：

Lstylel\=||1ClHlWl(Gl(y^)−Gl(y))||22

搞定了对风格的量化描述，接下来就要对图片的**内容**进行量化描述了，不过这个比较简单，直接用每一层的特征图来描述即可： {F1,F2,...Fl} ，相应的损失函数为

Lcontentl\=||1ClHlWl(Fl(y^)−Fl(y))||22

最后，为了保持风格转换后的低层的特征，还引入了两个简单的 Loss

Lpixel\=1CHW||y^−y||22

即耳熟能详的的 MSELoss。还有 total variation loss，这个 loss 的目的是为了提高图像的平滑度：

Ltv\=Mean||∑h,w(y^h+1,w−yh,w)+(y^h,w+1−yh,w)||22

整体的 Loss 就出来了：

L\=λstyleLstyle+λcontentLcontent+λpixelLpixel+λtvLtv

## 四、Perceptual Loss [Pytorch](https://zhida.zhihu.com/search?content_id=108596526&content_type=Article&match_order=1&q=Pytorch&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NjYyMjcxNTcsInEiOiJQeXRvcmNoIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTA4NTk2NTI2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.DswhFsB6Zd2R_v_PWAX-UwnC6hFfUujiaHfU7aVnQ6I&zhida_source=entity) 实现

网上的实现应该有不少，这里笔者提供一个十分简洁的实现：

```python3
__author__ = "charles"
__email__ = "charleschen2013@163.com"

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from style_transfer_perceptual_loss.image_dataset import get_transform
from src.utils.train_utils import get_device


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class PerceptualLoss:
    def __init__(self, args):
        self.content_layer = args.content_layer
        device = get_device(args)
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))
        style_image = Image.open(args.style_image).convert('RGB')
        _, transform = get_transform(args)
        style_image = transform(style_image).repeat(args.batch_size, 1, 1, 1).to(device)

        with torch.no_grad():
            self.style_features = self.vgg(style_image)
            self.style_gram = [gram(fmap) for fmap in self.style_features]
        pass

    def __call__(self, x, y_hat):
        b, c, h, w = x.shape
        y_content_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        recon = y_content_features[self.content_layer]
        recon_hat = y_hat_features[self.content_layer]
        L_content = self.mse(recon_hat, recon)

        y_hat_gram = [gram(fmap) for fmap in y_hat_features]
        L_style = 0
        for j in range(len(y_content_features)):
            _, c_l, h_l, w_l = y_hat_features[j].shape
            L_style += self.mse_sum(y_hat_gram[j], self.style_gram[j]) / float(c_l * h_l * w_l)

        L_pixel = self.mse(y_hat, x)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        L_tv = (diff_i + diff_j) / float(c * h * w)

        return L_content, L_style, L_pixel, L_tv
```

参考权重：pixel loss 权重与 content 一致

```text
parser.add_argument('--STYLE_WEIGHT', type=float, default=1e6, help='STYLE_WEIGHT')
parser.add_argument('--CONTENT_WEIGHT', type=float, default=1e0, help='CONTENT_WEIGHT')
parser.add_argument('--TV_WEIGHT', type=float, default=2e-2, help='TV_WEIGHT')
```

## 五、效果展示

PS：  
广告时间啦~  
理工狗不想被人文素养拖后腿？不妨关注微信公众号：  

欢迎扫码关注~