---
title:  "CIFAR-10 Estimator 学习和重写（一）：入口"
date:   2018-04-12
tags: ["学习笔记", "深度学习"]
abstract: >-
  对 TensorFlow 官方的 CIFAR-10 estimator 的学习和重写。
---

在 TensorFlow 的官方模型库中，有一个为 CIFAR-10 数据集实现的 ResNet Estimator 模型。在我使用 TensorFlow 的 Estimator API 实现了一个简单的 Vgg 模型之后，我决定去看一下那个 Estimator 的实现有什么高超的地方，以便来提高自己的开发水平。一看之下，还真是差别巨大。

[官方的 Estimator](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) 实现了以下高级的功能：

- 在单一的 CPU 上训练
- 在一个有多个 GPU 的机器上训练
- 在一个 CPU/GPU 集群上训练

通过在 CLI 指定运行的参数，就可以轻松转换训练的模式。不过，这套代码还是有一些问题，比如使用了 Experiment，没有使用 1.4 版引入的 `train_and_evaluation`。代码使用了一些兼容性的方式，我感觉也没什么用，毕竟 3.x 才是未来。所以我想在学习这套代码的基础上，使用最新的 API （现在是 1.7）来重写一下这个 Estimator。

先来点简单的，就是 `main.py` 的入口和 `main` 函数好了：

## 入口

`main.py` 的入口主要就是定义了 CLI 的参数：

```python
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
	  '--data-dir',
	  type=str,
	  required=True,
	  help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
	  '--job-dir',
	  type=str,
	  required=True,
	  help='The directory where the model will be stored.')
  parser.add_argument(
	  '--variable-strategy',
	  choices=['CPU', 'GPU'],
	  type=str,
	  default='CPU',
	  help='Where to locate variable operations')
  parser.add_argument(
	  '--num-gpus',
	  type=int,
	  default=1,
	  help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
	  '--num-layers',
	  type=int,
	  default=44,
	  help='The number of layers of the model.')
  parser.add_argument(
	  '--train-steps',
	  type=int,
	  default=80000,
	  help='The number of steps to use for training.')
  parser.add_argument(
	  '--train-batch-size',
	  type=int,
	  default=128,
	  help='Batch size for training.')
  parser.add_argument(
	  '--eval-batch-size',
	  type=int,
	  default=100,
	  help='Batch size for validation.')
  parser.add_argument(
	  '--momentum',
	  type=float,
	  default=0.9,
	  help='Momentum for MomentumOptimizer.')
  parser.add_argument(
	  '--weight-decay',
	  type=float,
	  default=2e-4,
	  help='Weight decay for convolutions.')
  parser.add_argument(
	  '--learning-rate',
	  type=float,
	  default=0.1,
	  help="""\
	  This is the inital learning rate value. The learning rate will decrease
	  during training. For more details check the model_fn implementation in
	  this file.\
	  """)
  parser.add_argument(
	  '--use-distortion-for-training',
	  type=bool,
	  default=True,
	  help='If doing image distortion for training.')
  parser.add_argument(
	  '--sync',
	  action='store_true',
	  default=False,
	  help="""\
	  If present when running in a distributed environment will run on sync mode.\
	  """)
  parser.add_argument(
	  '--num-intra-threads',
	  type=int,
	  default=0,
	  help="""\
	  Number of threads to use for intra-op parallelism. When training on CPU
	  set to 0 to have the system pick the appropriate number or alternatively
	  set it to the number of physical CPU cores.\
	  """)
  parser.add_argument(
	  '--num-inter-threads',
	  type=int,
	  default=0,
	  help="""\
	  Number of threads to use for inter-op parallelism. If set to 0, the
	  system will pick an appropriate number.\
	  """)
  parser.add_argument(
	  '--data-format',
	  type=str,
	  default=None,
	  help="""\
	  If not set, the data format best for the training device is used. 
	  Allowed values: channels_first (NCHW) channels_last (NHWC).\
	  """)
  parser.add_argument(
	  '--log-device-placement',
	  action='store_true',
	  default=False,
	  help='Whether to log device placement.')
  parser.add_argument(
	  '--batch-norm-decay',
	  type=float,
	  default=0.997,
	  help='Decay for batch norm.')
  parser.add_argument(
	  '--batch-norm-epsilon',
	  type=float,
	  default=1e-5,
	  help='Epsilon for batch norm.')
  args = parser.parse_args()

  if args.num_gpus > 0:
	assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
	raise ValueError(
		'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
	raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
					 '--variable-strategy=CPU.')
  if (args.num_layers - 2) % 6 != 0:
	raise ValueError('Invalid --num-layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
	raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
	raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

  main(**vars(args))
```

主要是定义了这么一些选项：

- `--data-dir` 是输入数据所在的地方
- `--job-dir` 这个也叫 `model-dir`，是 checkpoint 和 TensorBoard 数据存放的地方
- `--variable-strategy` 是指定参数服务器（parameter server），可以是 CPU（默认）也可以是 GPU
- `--num-gpus` 指定使用几个 GPU，默认是使用一个，如果只在 CPU 上运行，那就指定这个为 0.
- `--num-layers` 这个主要是指定 ResNet 的层数；我准备把这个去掉，因为我要把 `main.py` 做为一个通用的入口脚本，而指定层数并不是必须的。
- `--train-steps` 指定要训练多少次
- `--train-batch-size` 训练的 batch size
- `--eval-batch-size` 评估的时候的 batch size
- `--momentum` 指定学习中的“矩”
- `--weight-decay` 卷积参数的下降率
- `--learning-rate` 学习率。不过，这个学习率只有初始的学习率，在训练的过程中，学习率会在一定策略下下降，以增加学习的精度。
- `--use-distortion-for-training` 是否对输入的图片进行扭曲
- `--sync` 在集群训练中使用同步模式
- `--num-intra-threads` 使用几个线程运行训练/评估，如果指定为 0，那就会使用全部的 CPU
- `--data-format` 设定图像的数据格式，也就是 channel 在什么位置。可以是 channel_first，也可能是 channel_last。据说 channel_first 在 GPU 上更有效率，channel_last 在 CPU 上更有效率。
- `--log-device-placement` 是否显示任务分派的情况。
- `--batch-norm-decay` 设置 Batch Norm 的下降速率
- `--batch-norm-epsilon` batch norm 的另一个参数

接下来是对参数的一些检查，然后就可以调用 `main` 函数了。这里使用了一个参数展开：

```python
main(**vars(args))
```

我感觉没有必要。反正这些参数都是会直接传进去的。下面开始看一下 `main` 函数：

```python
def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement,
         num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = cifar10_utils.RunConfig(session_config=sess_config, model_dir=job_dir)
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(data_dir, num_gpus, variable_strategy,
                        use_distortion_for_training),
      run_config=config,
      hparams=tf.contrib.training.HParams(is_chief=config.is_chief,**hparams))
```

`main` 并没有做很多事情，主要的事情都分派给 `get_experiment_fu` 来完成了。不过，因为新的 API 使用了不同的启动方式，这个函数可能需要大改。

函数首先定义了两个默认的环境变量，这东西可能已经不再使用了，就不用管是什么意思了。然后创建了一个 `Session` 的 `config`，主要是指定了：

- 打开 `soft placement`
- 根据 CLI 选项指定是不是要输出设备的分派情况
- 根据 CLI 选项指定运行使用几个线程
- 打开 GPU 兼容模式

出于学习的目的，这里就要必要深入了解一下上面指定的这几个东西都是什么意思：

### soft placement

在 Programming Guide 的 [Using GPUs](https://www.tensorflow.org/programmers_guide/using_gpu) 中提到，如果想让 TensorFlow 在指定设备不存在的情况下，自动选择一个其它的设备来运行，那就可以设定 `allow_soft_placement` 为 `True`。只是这里不知道会不会影响效率什么的。

### log device placement

这个就比较好理解了。如果指定了上面的 `allow_soft_placement`，那怎么能知道最后是在什么设备上进行的计算呢？只需要指定 `log_device_placement` 为 `True` 就可以了。不过，因为是 log，我想可能会有点影响效率，也会增加 log 里的信息量，所以这里可以通过 CLI 的选项把它关掉。

### intra op parallelism threads

在 [爆栈](https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads) 上找到了一点说明。TensorFlow 中有两种形式的并行计算。如果一个操作本身是可以并行的，比如矩阵乘法，那么如果是在 CPU 上执行这类操作的时候，就可以使用一个 intra 的线程池，这个线程池的大小由 `intra_op_parallelism_threads` 指定。另外一种是两个不相关的操作（在计算图上没有依赖关系），会放到另一个线程池——inter 线程池——来执行。这两个参数如果指定为 0 ，那就由系统自动来设定。我想也就是设置为现在有的 CPU 核心数吧。不过，一般在 CPU 上也就是做一下可执行验证，然后就会扔给 GPU 去算了。不过，如果 CPU 被用来做参数服务器的话，那应该还是有用的。

### ConfigProto

在查找关于 `force_gpu_compatible` 的时候，找到了[这篇](https://www.jianshu.com/p/b9a442bcfd2e)介绍 `ConfigProto` 的文章。虽然只是对文档/注释的简单翻译，如果要了解一些配置项的意思，也很有参考价值了。

最后几行，因为是在构建一个 Experiment，我准备换了这个方式，就没有继续去研究了。

## `tf.estimator.train_and_evaluate`

在 TensorFlow 1.4 版中，引入了一个新的辅助函数：`tf.estimator.train_and_evaluate`，用来简化 Estimator 的训练、评估和模型导出。使用了这个函数，可以更方便的在 Google Clould ML Engine 上进行分布式训练。[文档](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate)里说，这个函数既可以支持单机的训练，又可以支持分布式的训练，应该算是对 `tensorflow/models` 里这个 CIFAR-10 的 ResNet Model Estimator 的另一种实现了。这样，`main` 函数就改成了：

```python
def main(args):
    # Session configuration.
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement,
        intra_op_parallelism_threads=args.num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    estimator = None
    train_spec = None
    eval_spec = None
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```

此时，当然还是不能运行的，estimator 好像容易实现一点，可以使用 `model_fn` 的方式创建，另外两个还不知道怎么创建。另外，在 `tensorflow/models` 的实现中，还实现了多 GPU 训练，以及使用 CPU 还是 GPU 做参数服务器（parameter server）的配置。这些都要在后面的学习中一步步实现。