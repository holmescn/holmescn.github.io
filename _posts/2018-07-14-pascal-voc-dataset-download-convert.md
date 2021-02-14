---
title:  "Pascal VOC 数据集下载与转换"
date:   2018-07-14
tags: ["研究笔记", "深度学习"]
abstract: >-
    Pascal VOC 数据集的下载与转换为 .tfrecords 文件的记录。
---

学习深度学习已经有一段时间了，一直在做单图像的分类。而实际上这个工作已经没什么太大的直接意义了。因为现在主流的分类网络已经可以达到一个非常了不起的糖度。另一方面，现实中的图像检测并不是在一个图像中只有一个目标物体这么简单的工作，通常都是在一个图像中有多个目标物体。这就提出了一个新的任务：不但要检测到图里有没有我们要找的目标物体，同时还要找到这个物体在图像中的位置。

本来，要做这个实验，我是要去使用 ImageNet 的数据的，还在 kaggle 上加入了 [ImageNet Object Detection Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)。不过，一下载数据集，我就傻眼了：图像数据集有 155 GB 之多。我想就算我下载下来了，能处理完一个 epoch，也是好几个月的时间了。于是，我就想起了在 ImageNet 之前非常流行的 PASCAL VOL 数据集。

## Pascal VOC 数据集简介

这里的 **Pascal** 不是 Pascal 语言，也不是帕斯卡这个人，而是一个缩写。全称是：*Pattern Analysis, Statistical Modeling and Computational Learning*，即“模式分析、统计建模和计算学习”。**VOC** 则是：*Visual Object Classes* 的缩写。

The PASCAL VOC project 包括了以下内容：

- 提供了一组标准化的图像数据集，用来做对象类型识别
- 提供了一组通用工具来访问这些数据和数据的标注
- 可以评估和对比不同的方法谁好谁坏
- 举办了一个挑战赛，来对比各个方法在对象类型识别上的性能（从 2005 年到 2012 年）

在 ImageNet 成熟之前，这个是最重要的计算机视觉领域的挑战赛之一。后来 ImangeNet 因为有更大的数据量，更多的类别，也就有了更好的通用懂，这个挑战赛才慢慢淡出了人们的视野。当然，和 [CIFAR-10/CIFAR-100](https://www.cifar.ca/) 数据集一样，做为深度学习的初学者，这个数据集还是很有用的。另外，虽然这个挑战赛已经不再继续了，但 PASCAL VOC 的评估服务器还一直在运行着。最新的消息（2018年7月）是 2014 年 11 月，评估服务器添加了一个对比不同模型的统计显著性的功能。

## Visual Object Classes Challenge 2012

最后一次挑战赛是在 2012 年举行的（ImageNet 是在 2017 年举行的最后一次挑战赛），本着有“有新不用旧”的想法，我决定选这个数据来做实验。原链接请点[这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

### 1. 赛事介绍

比赛的目标就是从实际的场景图像中识别出一些可见物体。这里有两个要点：1. 真实的场景；2. 可见的物体。真实的场景就是说，这不是一个简化的，一个图片只有一个物体，而且这个物体还正好在图像的中心，这样一个简单的识别任务。物体都是放在真实的场景中的，比如树下坐着一个人。另一个看起来是我故意加上去的，实际上同样重要的条件就是：可见的物体。我们人类在看见一个场景的时候，常常会想起这个场景中不存在的事物。比如在一个家庭聚会的场景中，想起已经去世的亲人之类的。而这些目前还不是计算机视觉可以解决的问题。

这个数据集一共有 20 个小类，分别归在 4 个大类中：

- 人
- 动物：鸟、猫、牛、狗、马、羊
- 交通工具：飞机、自行车、船、公交车、小汽车、摩托车、火车
- 居家用品：瓶子、椅子、餐桌、盆栽、沙发、电视

具体任务有

- 分类：给一个图片，确定上面 20 个小类里，那些存在，哪些不存在
- 探测：找到以上 20 分类中存在的物体的边框、以及对应边框的类别
- 扣图：不只是图出目标物体的边框，还要以像素级的精度画出图片中的物体
- 动作分类：预测图像里的人在干什么
- Person Layout：找到人的头、手、脚等的边框

### 2. 数据描述

训练数据是由一些图片，以及对应每个图片有一个标注文件。标注文件包含了这个图像里出现的每个物体的边框和类别。需要注意的是，并不是每种类型的对象就只出现一次。每个类型的物体都可能出现多次。

还有一小部分图片有详细的扣图的标注，这样就可以做像素级的扣图了。当然对于其它两种任务，也都有对应的数据。

关于数据的细节，等做实验的时候再说吧。

## 数据下载与转换

本来数据集应试有三个部分

1. train
2. validation
3. test

因为我们也不准备参加 Pascal VOC 的竞赛，所以 test 的部分就可以忽略掉了。下载地址请点[这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)。

### 1. 下载步骤

先创建一个 folder 用来下载。当然也可以在浏览器里直接点击下载。

    $ mkdir ~/data
    $ cd ~/data

我这里就用 macOS 系统演示了。使用 Linux 的同学基本可以无缝操作。使用 Windows 的同学去面壁。然后开始下载：

    $ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

第一个文件大概有 2 GB，所以如果网速慢，就多等一下。下载完成以后，开始解压：

    $ tar xvf VOCtrainval_11-May-2012.tar

解压完，会出现一个 `VOCdevkit` 的 folder（因为不知道应该叫“目录”还是叫“文件夹”，所以先用英语）。里面有一个 `VOC2012` 的 folder。其中包含：

1. Annotations
2. ImageSets
3. JPEGImages
4. SegmentationClass
5. SegmentationObject

其中的 `JPEGImages` 是所有的图片，其它都是对图片的标注。看了一下，一共有 17000 多张图。当然每张图里可能有多个对象。即使是这样，这个数据集和 ImageNet 比起来也是小太多了。另外，因为图像都是从实际场景中取出来的，所以每张图像的大小都不一样，这个也是在构造模型的时候应该考虑的事情。

对于分类/探测类的任务，一共提供了 4 个数据集：

1. train：训练集
2. val：验证集
3. trainval：训练集和验证集的并集
4. test：这个没有

### 2. 转换为 .tfrecords

在完成了下载以后，我要把这些数据转换成一个 `.tfrecords` 文件。这样就可以使用 TensorFlow 的 DataSet API 来使用了。我的目标是做分类/探测，所以需要同时保存一个图片里有哪些对象，以及每个图像的边框。等不及直接看源码的同学，请直奔[这里](https://gist.github.com/holmescn/a7712cbac8f4b54fe3dca0decfce06fe)

TFRecord 把每一条训练数据都组织成一个 example，这里，我把 example 的结构定义为：

<code data-gist-id="a7712cbac8f4b54fe3dca0decfce06fe" data-gist-line="53-63"></code>

其中，`image` 对应一副输入的图像。因为 TFRecord 并不保留数据的原始形状，所以还需要把图像的 shape 保存起来，以便在后面读取图像的时候使用。原来我是准备把 `objects` 实现成一个 one-hot 型的数据，但想想那样还需要同步类型的排列，所以最后 `objects` 就是图像中出现的 object 的名字。这样在训练的时候，可以自己决定使用什么样的方式来处理分类的数据。最后是 bounding box，受限于 TFRecord 的表达难力，没有办法生成类似 `[[xmin, xmax, ymin, ymax], [xmin, xmax, ymin, ymax], ...]` 的形式，就算实现了这种结构，使用的时候也要麻烦一些。所以直接把 bbox 展开了，使用了 4 个 `Int64List`，这样所有 `index=0` 的都是同一个 object 的边框了。更多代码细节，可以到 [Gist](https://gist.github.com/holmescn/a7712cbac8f4b54fe3dca0decfce06fe) 里去看。这样，明天就可以开始做 SSD 了。

