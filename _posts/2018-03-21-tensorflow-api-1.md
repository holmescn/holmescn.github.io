---
title:  "TensorFlow API 学习（一）"
date:   2018-03-21
tags: ["学习笔记", "深度学习"]
abstract: >-
    TensorFlow API 学习笔记（一）。
---

TensorFlow 目前（2018 年 3 月）显然是深度学习的主流框架，根据
[一些统计结果](https://zhuanlan.zhihu.com/p/34500929)也可以证实这一点。在我写这篇文章的时候，使用 `pip` 安装的 TensorFlow 还是 1.6 版，但实际上 1.7 版已经发布了。对于快速发展的行业来说，一个小版本号的差异也可能是一个巨大的进步。不过，因为在线的 API 文档还停在 1.6 版，而[已经有人总结了1.7版的变化](http://www.xnb.cn/wechatarticle-8341.html)，可能有一些小变化，好像没有巨变的突破，通过这篇文章了解一下就可以了。

tf 的文档（Python）分为三个部分：Modules，Classes 和 Functions。因为刚开始研究，所以还不能准备归纳每个部分的内容。大致看了一下，可能是这样的：Classes 中定义了 Functions 里参数和返回值的类型，而 Modules 应该是把一些同一功能组函数和类放在了一起。现在不能确定的是，Functions 里是不是有一些是 Modules 里的功能的简单封装和快捷函数。下面先从 Modules 开始，梳理一下 TensorFlow 的功能：


## tf.app

这个模块相当于为 TensorFlow 进行的脚本撮供了一个 main 函数入口，可以定义脚本运行的 flags。这就为 TensorFlow 的训练和评估脚本提供了一个统一的框架，而不用每个脚本都自己去拼凑一套程序架构了。不过，我是感觉 `argparse` 这个包的定义方式好像更好一点，`tf.app` 的定义方式，并不是很习惯。这可能也是因为 TensorFlow 底层是 C++ 写的，Python 只是一个 SWIG 的壳吧？这个还不能确定。

## tf.bitwise

这个模块提供了一组*位操作*。这里的**操作**并不是 numpy 的那种 forward 到 C++ 的加速，而是创建一个计算图的节点。不过，这东西有没有 GPU 加速也不知道。提供的操作就是：与、或、非、异或、左移和右移。

## tf.compat

这个模块提供了一组函数来解决 Python 2.x 和 3.x 的兼容问题。主要就是类型的转换函数。我感觉 Python 3.x 一定是未来了，如果要运行蹦遗留代码，不如直接重写好了。反正一般代码也不长。

## tf.contrib

这个模块可以当成是一个实验场，主要是针对 TensorFlow 的改进的测试，我想一些成熟的尝试最后就会进入到主线，一些失败的尝试可能就被删了。所以这里应该会有一些黑科技吧。

## tf.data

这个应该是 TensorFlow 高级 API 的一部分，主要是提供了输入数据的处理功能。我猜想，相比使用 `ndarray` 做输入数据和 mini-batch，这个模块应该提供了一些更方便和高级的函数和方法。这样我就不需要费力去使用 pandas 或者 array 来自己组织数据了。应该是一个很方便的输入数据的组织方式。

## tf.distributions

这个模块提供了一组随机分布。文档说这个是 TensorFlow 的核心模块之一。我猜想这个模块首先是用来初始化权重的值。但它还有没有别的用处我还没想到。可能在后面做具体模型的时候才能体会到吧。

## tf.estimator

初看这个模块，我还以为只是在 `scikit-learn` 的 API 上，实现了分类和回归分析。在初步了解了一点高级 API 之后，才发现这个其实是一个高级概念的集合。一个 Estimator 其实是一个 Model + Training + Evaluate 的合体。在模块中，已经实现了几种简单的分类器和回归器，包括：Baseline，Learning 和 DNN。还有一些组合版的。不过，这里的 DNN 的网络，只是全连接网络，好像没有提供卷积层之类的。不过，复杂的模型可以通过 `Estimator` 这个类直接实现。下面是两篇讲解高级 API 的文章，好像官网的文档也有关于高级 API 的文档：

- [如何使用TensorFlow中的高级API：Estimator、Experiment和Dataset](https://www.jiqizhixin.com/articles/2017090901)
- [Higher-Level API in TensorFlow](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0)

## tf.feature_column

文档说，这个模块是用来提取和表示特征的。这个还不能很好的理解。简单看，好像就是把一些特征向量化。应该也是提供了一组方便的数据预处理工具。不知道是高级的 API 还是底层的 API。

## tf.gfile

这个模块提供了一组文件操作函数。本来 Python 已经有一组还不错的文件操作函数了，我想这个只是对 C++ 的封装？也可能是为了保持代码的一致性，或者是为那些对 Python 并不是很熟的用户提供一组文件操作？或者是这一组文件操作的效率更好一些？不管是因为什么吧，反正这一组 API 就是操作文件的。

## tf.graph_util

这个模块的描述是说在 Python 里直接操作 tensor 图。提供的函数并不多，可能是在一些高级模型中才有需要吧。

## tf.image

TensorFlow 的图像处理操作。这些操作应该也是要添加到计算图上的，不是 skimage 那种直接处理图像的操作。主要是一些颜色变换、变形和图像的编码和解码。

## tf.initializeers

主要是用来生成一些初始化了的 tensor。比如常见的 `truncated_normal` 之类的。这个如果是自己写模型，可能是很有用的。不过，不知道高级 API 是不是也需要这些。

## tf.keras

Keras 本来是一个独立的深度学习库，可以更改后端，选择 TensorFlow，Theano 等。不知道 TensorFlow 内部实现了这一套 API
是什么意思？有可能是因为这套 API 太好用了，所以直接被吸收了？也可能是 Keras 的维护进度跟不上 TensorFlow 的更新速度？
因为我单独安装了 Keras，依赖的 TensorFlow 版本是 1.0。大概看了一下模块的内容，应该就是把 Keras 的 API 直接拿了过来，
因为 Keras 的 API 是更加 High-Level 的，所以这个模块应该在构建经典网线的时候很方便。只是，这已经是另一套 API 了。

## tf.layers

高级 API，不是通过 variable 的方式来定义一个层，而是以更高级的概念——层——来定义一个模型。看起来有些像从 Keras 里抄过来的样子。不过应该挺好用的。常用的池化层、卷积层什么的都有。

## tf.linalg

线性代数操作，比如求对角化，求逆什么的。应该也是添加到计算图中的操作。所以这个也是比较底层的操作了。

## tf.losses

一些常用的损失函数。也是方便定义模型用的。这样就不用自己用 Python 写损失函数了。毕竟常用的损失函数也就那么几个。要自己定义损失函数，不如把解空间映射到常用损失函数上。

## tf.manip

操作 tensor 的操作。不过没什么内容，只有一个旋转 tensor 的操作。不过也是添加到计算图的操作。

## tf.metrics

这个和 `tf.losses` 模块一样，是提供了一些在测试时常用的评估函数。比如预测精度之类的。这些也是方便了构建网络而提供的辅助函数，和上面的观点一样：尽量把求解的问题真的成这些函数可以描述的解空间，而不是用一个看起来直接，但不好评估的新函数。

## tf.nn

这个模块提供了一些构建神经网络的底层函数。应该 TensorFlow 构建网络的核心模块了。其中包含了添加各种层的函数，比如添加卷积层、池化层等。

## tf.python_io

看到这个模块的名字，我还以为是封装了什么 I/O 函数，看了一下，是用来读写 `TFRecord` 的。现在还不能确定 `TFRecord` 是干什么的。是保存网络模型的还是训练数据的。

## tf.resource_loader

一组读入资源文件的函数，但还不知道这里说的*资源*是什么东西。

## tf.saved_model

这个模块功能很确定，就是保存和恢复一个模型的。所以上面那个 `tf.python_io` 应该就不是保存模型的了。

## tf.sets

TensorFlow 提供的一组*集合*的操作。看目录结构，应该是计算图上的操作，而不是直接得到计算结果的函数。

## tf.spectral

一些谱变换操作，添加到计算图上的。比如 FFT，DCT 什么的。

## tf.summary

用来生成 TensorBoard 可用的统计日志。目前 Summary 提供了 4 种类型：audio、image、histogram、scalar。还有一个 text，不知道是不是也是一种。

## tf.test

提供了一些单元测试的功能。可以测试每个计算节点的输出？应该对实现新网络层有好处吧。目前我能实现的网络，都是常见的网络，所以应该也没什么要测试的需求。

## tf.train

这个模块提供了一些训练器。应该是和 `tf.nn` 组合起来，现实一些网络的计算。不过，在实现 DNN 的时候，应该是使用高级 API 更多一些，使用这些底层 API 的好处就是能更灵活的构建一些新的网络结构，甚至一些非神经网络的计算。

## misc

还有一些模块，用处不是很大：

- `tf.errors`：定义了 TensorFlow 的错误信息
- `tf.flags`：和 `tf.app.flags` 是一个东西，就是在 `tf.app` 里实现一些参数
- `tf.logging`：日志
- `tf.profiler`：性能监测器
- `tf.pywrap_tensorflow`：文档里没内容，是 SWIG 的封装操作，一般也不会用到
- `tf.sysconfig`：获取一些系统配置，比如编译选项什么的
- `tf.tools`：文档里没内容
- `tf.user_ops`：应该是用户定义的计算图操作，但没有内容

## 结语

因为刚开始学习时间不长，有很多东西都理解不深。后面学习深入之后，再补充一些内容吧。
