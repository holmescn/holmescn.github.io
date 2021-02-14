---
title:  "TensorFlow 高阶 API 学习笔记"
date:   2018-03-27
tags: ["学习笔记", "深度学习"]
abstract: >-
    TensorFlow 高阶 API 学习笔记。
---

在学习 TensorFlow 的时候，先浏览了 API 的模块划分，发现了 `tf.estimator` 这个模块。进一步研究发现，TensorFlow 已经实现了一套高阶的 API 用来简化模型的实现。这也让我之前根据 CIFAR10 模型创建的 scafold 变的没有意义。于是找了两篇文章，想要学习一下 TensorFlow 的高阶 API。可是，在读完了 [这篇](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0)（[译文](https://www.jiqizhixin.com/articles/2017090901)）介绍文章之后，感觉不是很到位。因为文章的内容好像是基于 1.3 版的 API 写的，那个时候，很多的 API 还都在 contrib 里。现在已经是 1.7 版正式发布了，可见 TensorFlow 发展的速度真是一日千里。在看 Get Started 的时候，感觉明矾描述的 Estimator 的定义方式都和这篇文章写的不太一样，那里需要通过继承 `tf.estimator.Estimator` 来实现新的模型，而这篇文章只需要提供 model function 就可以了。所以我还是直接去看在线文档好了。

## Dataset

Dataset API 主要是完成在 set 这个层级的对数据的操作，比如 shuffle，repeat，map 等。入门的使用会用到
`tf.data.Dataset.from_tensor_slices` 这个函数。这个函数是把输入沿着指定的一个方向进行 slice 操作，
比如对于 MNIST 的 `(N, 28, 28)` 这样一个形状的输入，类似于：

    dataset = [data[i] for i in range(N)]

最后得到 N 个 dataset，每个是一个 `(28, 28)` 的形状。如果输入的是 `dict` 类型，那就有意思了：

    ds = tf.data.Dataset.from_tensor_slices({
        "a": [1, 2, 3, 4],
        "b": [2, 3, 4, 5]
    })

会得到这样一组 `dataset`:

    [{
        "a": 1,
        "b": 2
    }, {
        "a": 2,
        "b": 3
    }, ...]

类似的，也可以使用 `tuple`。

我感觉 `tf.data.Dataset.from_tensor_slices` 对于  CNN 等大型的网络没什么用。因为训练这些网络都需要很大的数据集，
这些数据集都不可能直接保存在内存里。同时，因为这个操作会把所有的数据使用 `tf.constant` 包起来，放到计算图里，如果数据量太大，
这就会生成一个很大的计算图，效率就很低了。（因为数据会被复制多次，还有 2GB 的限制）

`Dataset` 可以对一个数据集进行抽象的操作：

- `shuffle` 用来做数据顺序随机化，减少对数据顺序的依赖
- `repeat` 让数据集可以重复
- `batch` 把数据集切分成 batch
- `map` 类似函数式语言的 map 操作，就是把一个数据点转换成别的东西。

> The train, evaluate, and predict methods of every Estimator require input functions to return a (features, label) pair containing tensorflow tensors.

也就是说，input function 返回是一个 `tuple`，形式是： `(features, label)`。而访问 Dataset 的方式应该是使用 `iterator`。
Dataset 支持很多种 `iterator`，常用的有：`one shot iterator` 和 `initializable iterator`，还有别的，但好像都不太常用。
`one shot iterator` 最简单，只需要使用 `make_one_shot_iterator` 创建一下就可以了，一次获取一个数据点。`initializable iterator`
则是在迭代之前，需要提供一个参数。

在 Quick Start 的后面，讲了怎么读入 CSV 文件，但我并不需要这样的东西。我需要的是读 TFRecord 和图像。

`tf.data` 引入了两个抽象：

1. `tf.data.Dataset` 表示一系列的元素，每个元素是一个或者多个 Tensor。有两种方式创建一个 Dataset：
    1. 创建一个数据源，比如 `tf.data.Dataset.from_tensor_slices`
    2. 应用一次变换，比如 `tf.data.Dataset.map`
2. `tf.data.Iterator` 用来从一个 dataset 中获取数据

一个 dataset 包含多个元素，每个元素都有相同的结构。一个元素又包含一个或多个 `tf.Tensor`，称为*组件*。每个组件有一个 `tf.DTye`，代表
这个组件的数据类型（`tf.Tensor` 里每个数据的类型）；还有一个 `tf.TensorShape`，代表每个元素的静态 shape。`Dataset.putput_types` 和 `Dataset.putput_shapes` 保存了推断出来的每个组件的类型和 shape。

dataset 提供了很多的变换操作：

- `map`
- `flat_map`
- `filter`
- ...

除了上面说的 `one_show_iterator` 和 `initializable_iterator`，还有两个：

`reinitializable_iterator`，通过 Dataset 的 structure 来创建，可以绑定到同一种结构的不同 dataset 上。比如有两个 dataset，
一个增加了噪声，用来做训练，一个没有噪声，只用来评估结果。那就可以创建一个 `reinitializable_iterator`，实现两个数据集之差的切换。

`feedable_iterator` 刚是通过一个 `placeholder` 来控制在每次运行的时候，要使用哪种 iterator 来获取数据。

`tf.data.TFRecordDataset` 说是提供了一种处理大数据集的方法。也就是说 TFRecord 文件可以很大，很多，但并不影响使用。
如果输入是文本文件，格式是分行的，那就可以使用 `tf.data.TextLineDataset` ，这个一次传回一行，再通过 dataset 的变换函数，就可以实现
丰富的数据输入源了。比如使用 CSV 文件模式。

## Estimators

Estimator 是高级 API 的核心，整合了：

- 模型定义
- 训练
- 评估
- 预测
- 导出

优点：

- 无需修改就可以迁移到集群、TPU等高级设备，只需要在本地可以测试就行了
- 简化开发者之前的共享实现
- 更容易创建新模型
- Estimator 是基于 `tf.layers` 的，所以很容易定制
- 安全的分布式训练循环，可以控制：
    - 什么时候/如何 构建计算图
    - 什么时候/如何 初始化变量
    - 什么时候/如何 开始队列
    - 什么时候/如何 处理异常
    - 什么时候/如何 创建 checkpoints 文件和从错误中恢复
    - 什么时候/如何 保存 TensorBoard 数据

在 Estimator 里关于 pre-made Estimator 的使用说明中，文档说，一个 input function 要返回以下两个东西：

1. feature_dict，keys 是 feature 的名字，value 是对应的数据，使用 Tensor
2. labels，使用 Tensor

也就是说，features 的格式是有限制的，不能随便写，当然这个是使用 pre-made 的 Estimator 的说明，后面说到自定义的 Estimator
不知道这个限制有没有改变。

在定义完 Input function，之后，还要定义 feature column。创建 Estimator 的时候，要传入这些 feature column，然后，在训练的时候，
办要传入，

文档里推荐了一个工作流程：

1. 先用 Pre-made 的 Estimator 调试 Dataset，保证输入数据流的正确性等问题，并以这个工作做为一个基准
2. 定义自己的模型

TensorFlow 提供了一种方法把 Keras 的模型转换成 Estimator，但这不是我想要的。就略过去吧。

定义 model function 的样子是：

    def my_model_fn(
        features, # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params):  # Additional configuration

要完成一个自定义 Estimator，要完成下面两项工作：

- 定义模型
- 定义对应下面状态下的其它计算：
    - Predict
    - Evaluate
    - Train

定义模型，就是要定义：1. 一个输入层；2. 一些中间层；3. 一个输出层

通常，输入层是通过 feature_dict 和 feature column 来创建的：

    # Use `input_layer` to apply the feature columns.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

然后，这里只创建一些全连接层：

    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

显然，使用 Layer 这个抽象，可以避免自己创建 variables 和 biases 的底层数据类型。最后，定义输出层：

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

下面就需要针对模型的状态来添加其它的计算了。这里要说明的就是，每个 model function 最后都要返回一个
EstimatorSpec 实例：

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                predictions=predicted_classes,
                                name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

这样，一个 model function 就创建完了。然后是怎么用：

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

在使用上，pre-made Estimator 和 custom Estimator 都是一样的。只有怎么传参数的区别。
