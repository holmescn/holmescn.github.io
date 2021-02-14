---
title:  "CIFAR-10 Estimator（二）：MNIST 的 CNN 模型"
date:   2018-03-30
tags: ["学习笔记", "深度学习"]
abstract: >-
    使用 TensorFlow 的高级 API （Dataset, Estimator, Layer, Metrics）实现的 CIFAR-10 卷积神经网络。
---

初学 TensorFlow 的时候，看的是极客学院翻译的文档，里面讲 CNN 模型的时候，使用的还是底层 API 实现的。本来我还把这套模板抽象成了一个 scafold，以方便后面的开发。但进一步研究之后，发现新的 TF 已经支持一系列的高级 API 了，比如 Estimator/Layer/Metrics，还有 Keras 的 API。于是就想把那套 CIFAR-10 的模型用 Estimator 再实现一下。

经过初步尝试之后，发现里面还是有一些问题目前理解不了。比如怎么在 layers 里实现 weight decay。所以又去看官方的文档，发现新的教程已经在使用 layers 实现 CNN 了。虽然教程里的数据集是使用的 MNIST。正好 CIFAR-10 的分类数也是 10，所以我准备先用这个教程实现一个 CNN，移花接木来处理一下 CIFAR-10 的数据，同时进一步熟悉一下 Estimator 的编写流程。等对低级 API 有了更多的了解，再去实现之前的 CNN 模型。

## 程序结构

TensorFlow 的 Estimator 封装了训练、验证和预测的基本流程。同时，把数据预处理的部分使用 Dataset 分离了出来。这样，一方面可以在同一个模型上测试不同的数据集，又可以很方便的测试不同模型的预测效果。这里，我为了达到尝试不同 CNN 模型的效果，把程序进行了一些拆分：

```python
class CNNEstimator(tf.estimator.Estimator):
    def __init__(self, model_dir=None, params=None, config=None):
        super().__init(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params
        )
    
    def model_fn(self, features, labels, mode, params):
        pass
```

好像做深度学习的同学都不是很需要 OOP 的编程方式，看到的一些教程也都是基于 module 里的函数定义式的方法。这种方法在写教程的时候，会避免一些不必要的程序结构。不过，我感觉还是使用一点“类”的概念，把每个 Estimator单独出来，就像 TensorFlow 里那些 premade 的 Estimator 一样，用起来更舒服一些。所以，我没有使用 Estimator 构造函数传入 model_fn 的方式，而是继承了 `tf.estimator.Estimator`，然后在初始化的时候，传了一个 method 进去。这个 method 当然就是标准的 model_fn。

## `model_fn`

下面定义 `model_fn`：

```python
def model_fn(self, features, labels, mode, params):
    # Use `input_layer` to apply the feature columns.
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])

    logits = self.construct_model(input_layer, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

根据 [Creating Custom Estimators](https://www.tensorflow.org/get_started/custom_estimators) 里创建 `model_fn` 的方法，从 Dataset 创建一个输入层的方法，就是使用 `tf.feature_column.input_layer` 来创建。配合传入的 `params`，从输入的数据里生成一个输入层。本来这一行，我是放在 `construct_model` 这个函数里的，但观察了一下这个函数，其实对 `features`、`labels` 什么的并没有什么依赖关系，就把它提出来了。

这里我进一步把构建模型的过程分离了出去。因为其它的根据模式定义 `loss`、`metrics` 等，都相对固定，除非使用了什么特别的模型，一般不用修改。而最常变动的应该就是构建模型的部分了。

```python
def construct_model(self, input_layer, is_training):
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits

```

最后就是构建网络了。这个卷积网络很简单，可以也是之前教程中那个双卷积层网络的一种实现，只是没有看到关于 weight decay 的方法。这里，需要对网络的细节时行一点调整。因为 MNIST 的图片是 28 x 28 的，所以经过两次 pool 之后，会变成 7 x 7 的小图。而 CIFAR-10 的图片是 32 x 32 的，经过两次 pool 之后，就变成了 8 x 8 的小图。所以要对 `pool2_flat` 的维度进行一些调整。

下一篇再说怎么使用这个 Estimator。
