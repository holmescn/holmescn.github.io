---
title:  "CIFAR-10 Estimator（三）：训练"
date:   2018-04-02
tags: ["学习笔记", "深度学习"]
abstract: >-
    使用 TensorFlow 的高级 API （Dataset, Estimator, Layer, Metrics）实现的 CIFAR-10 卷积神经网络。
---

在完成了[输入数据转换](http://www.holmesconan.me/2018/03/cifar10-estimator-1.html)和[模型构建](http://www.holmesconan.me/2018/03/cifar10-estimator-2.html)之后，终于要进入最后一步了，就是模型的训练、验证和预测。对于 Estimator 来说，这个过程是固定了，所以我单独写了一个文件：`main.py`。先从脚本的入口开始看吧：

```python
if __name__ == '__main__':
    tf.app.flags.DEFINE_string("model_dir",
                                "/tmp/cifar10_model",
                                "Specify the model dir.")
    tf.app.flags.DEFINE_string("data_dir",
                                "/tmp/cifar10_data",
                                "Specify the data dir.")
    tf.app.flags.DEFINE_string("mode",
                                "train",
                                "Runing mode.")
    tf.app.flags.DEFINE_float("learning_rate",
                                0.001,
                                "Learning rate.")
    tf.app.run()
```

这里，我使用了 `tf.app.run()` 这个方法，而没有使用 `argparse`，我感觉这没什么不同，只是让程序更一致而已。这里我定义了一些命令行的选项，这样方便在不修改程序的情况下，对参数进行测试。而程序的真正入口在：

```python
def main(unused_argv):
    estimator = CNNEstimator({
        'feature_columns': [tf.feature_column.numeric_column('image')],
    })

    if FLAGS.mode.lower() == 'train':
        estimator.train(input_fn=lambda: input_fn(os.path.join(FLAGS.data_dir, TRAIN_FILE)))
    elif FLAGS.mode.lower() == 'validation':
        estimator.evaluate(input_fn=lambda: input_fn(os.path.join(FLAGS.data_dir, VALIDATION_FILE)))
    elif FLAGS.mode.lower() == 'predict':
        estimator.predict(input_fn=lambda: input_fn(os.path.join(FLAGS.data_dir, EVAL_FILE)))
    else:
        print("Unknown mode: %s" % FLAGS.mode)
```

这里的 `CNNEstimator` 是我们自己创建的那个 Estimator。不过，传入的参数现在并没有使用，因为使用 `feature_column` 需要使用一个 `feature_dict`，而这个 `dict` 的使用现在有点问题，就先略过去了。

创建完成之后，就是根据运行的 `mode` 来选择怎么使用这个 Estimator 了。不过，无论是哪个方法，都需要传入一个 `input_fn`：

```python
def input_fn(filenames):
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(parser)  # Parse the record into tensors.
    dataset = dataset.shuffle(buffer_size=1000) # Shuffle the dataset
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(32)

    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels
```

这里就用到了 Dataset 的 API 了。先使用 `TFRecordDataset` 创建一个 dataset，然后通过 `map` 来解析，再对数据进行 `shuffle`、`repeat` 和 `batch`。最后，要获取 dataset 里数据，还需要创建一个 Iterator，再调用 iterator 的 `get_next()` 就可以了。这里为了简单，我使用了 `one_shot_iterator`，也就是一次获取一个数据点的 `iterator`。那么，要怎么解析呢：

```python
def parser(record):
    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label
```

代码简单、直接，好像也没什么要解释的。这样，就可以用这个脚本来训练模型了。完整的代码[请点这里](https://github.com/holmescn/deep-learning-practice/blob/master/tensorflow/estimators/cifar10/main.py)。

然后，就可以使用下面的命令来开始训练了：

    $ python main.py

只是记得在训练之前要完成前面两个步骤：转换数据和构建模型。
