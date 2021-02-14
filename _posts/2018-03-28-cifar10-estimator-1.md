---
title:  "CIFAR-10 Estimator（一）：数据转换"
date:   2018-03-28
tags: ["学习笔记", "深度学习"]
abstract: >-
    使用 TensorFlow 的高级 API （Dataset, Estimator, Layer, Matrics）实现的 CIFAR-10 卷积神经网络。
---

学习了三天 TensorFlow 的高级 API，终于把创建自定义的 Estimator、Dataset、Feature column 这些
东西都学了一遍。今天准备牛刀小试，使用高级 API 来完成教程里的 CIFAR-10 卷积神经网络。网络结构和一些
基本信息，需要参考 [Github](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10) 上的源码。
在 TensorFlow 的 models 项目里，也有一个 [CIFAR-10 的 Estimator 实现](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)，
作为练习，我不准备参考这个实现的任何代码，不过，其中提供的一些结构，对于创建一个基于 Estimator 模型
还是很有参考意义的，所以我准备模仿这个实现的文件结构。

## 生成 TFRecord 文件

CIFAR-10 的原始文件是使用一个 bin 的模式保存的，包含了一些图片和图片对应的 label。而 TFRecrod
是 TensorFlow 推荐使用的数据格式。在 Dataset API 里还专门有一个 TFRecrodDataset 用来读入 TFRecrod 文件，优点就是：自动处理文件队列，不用把所有的数据都存放在内存里。所以，为了利用这个 Dataset，就需要先把 CIFAR-10 的原始文件转换成 TFRecord 文件。也就是 `cifar10_estimator` 里的 `generate_cifar10_tfrecords.py` 这个文件。关于怎么读写 TFRecord 文件，我找到了[这篇](https://www.skcript.com/svr/why-every-tensorflow-developer-should-know-about-tfrecord/) 文章。当然，因为这个过程也没什么内容，我就直接把官方实现改一下：

下面逐步分析一下，当然是从入口开始：

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/cifar10_data',
        help='Directory to download and extract CIFAR-10 to.')

    args = parser.parse_args()
    main(args.data_dir)
```

这个下载转换脚本，通过 `argparser` 定义了一个选项：`--data-dir`。这样，就可以这样使用：

    $ python generate_tfrecords.py --data-dir "/tmp/another-path"

默认我写了一个地方来保存下载的数据。下面，进入 `main` 函数：

```python
def main(data_dir):
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')

        if os.path.exists(output_file):
            os.remove(output_file)

        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)

    print('Done!')
```

`main` 函数的流程很直接，模块划分的也很好。先从网上下载 CIFAR-10 的数据，是一个 `.tar.gz` 的文件。然后解压到指定的 `data_dir` 里。下载好的函数是分成几个 batch 的，这里通过手工的方式来划分开训练、评估和测试。得到一组输入数据的文件名。然后开始处理这些数据：先把文件名和路径合一，然后删除之前生成的文件，再把 CIFAR-10 的数据文件转换成 `.tfrecords` 文件。原版在删除 `.tfrecords` 文件的时候，使用的是一个异常处理的方法。我感觉这个方法并不方便。如果文件不存在，那就跳过好了。可如果是因为没有权限，那这个异常就被跳过了，后面就可能出错。所以我把这个改了一下。

本来我是使用 `urllib` 来下载数据的，不过，感觉这样没有进度数据，用起来不是很方便，于是我就用 shell 脚本写了一个下载解压的脚本：

```bash
#!/bin/sh
CIFAR_FILENAME='cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL="https://www.cs.toronto.edu/~kriz/$CIFAR_FILENAME"
DATA_DIR='/tmp/cifar_data'

mkdir -p $DATA_DIR

OLD_PWD=$PWD
cd $DATA_DIR
wget $CIFAR_DOWNLOAD_URL
tar -xvzf $CIFAR_FILENAME
cd $OLD_PWD
```

这里我使用了 `wget` 来下载，当然也可以使用 `curl`，因为功能很专一，并没有写得很复杂，也没有使用参数来指定目标路径。整个数据下载有 160+MB，所以还是要保存好了，省得每次都要下载。这样，我就可以把 python 里的下载函数删除了：

```python
def main(data_dir):
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    # download_and_extract(data_dir)
    file_names = _get_file_names()
    ...
```

解压后的数据文件一共有 6 个 batch，分为 5 个数据 batch，1 个测试 batch。接下来把这几个 batch 分为三组，分别用来做训练、验证和评估：

```python
def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names
```

也就是 1 至 4 的 batch 用来训练，第 5 个 batch 用来验证，测试 batch 用来评估。最后会生成三个 `.tfrecords` 文件：`train.tfrecords`，`validation.tfrecords` 和 `eval.tfrecords`。

最后是核心函数：`convert_to_tfrecord`：

```python
def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = _entry2example(data[i], labels[i])
                record_writer.write(example.SerializeToString())
```

要输出 `.tfrecords` 文件，首先需要一个 `tf.python_io.TFRecordWriter`，使用 `with` 语法，简化了对文件操作过程中的问题。这里下载的版本，是使用 python 的 `pickle` 序列化的，只需要使用 `pickle` 来反序列化一下就可以了：

```python
def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict
```

这里，因为使用 python3，遇到了第一个坑：pickle 要使用 python3 自己的 `pickle` 包，并指定 encoding 为 bytes。另一个坑是：因为使用了 encoding='bytes'，`data_dict` 的 keys 都变成了 bytes，这也需要改过来。接下来，需要把每对数据转换成一个 `tf.train.Example`：

```python
def _entry2example(data, label):
    return tf.train.Example(features=tf.train.Features(
        feature={
            'image': _bytes_feature(data.tobytes()),
            'label': _int64_feature(label)
        }))
```

还有两个类型转换：

```python
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
```

对于以图像为输入的 CNN 网络来说，这些辅助函数都很常用。

这样，整个转换的过程就完成了。完整的代码请[点击这里](https://github.com/holmescn/deep-learning-practice/blob/master/tensorflow/estimators/cifar10/convert_to_tfrecords.py)。
