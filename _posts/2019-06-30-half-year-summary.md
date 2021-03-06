---
title:  "2019 年上半年总结"
date:   2019-06-30
tags: ["年中总结", "2019"]
abstract: >-
    忙忙碌碌，2019 年就过去了一半了，回头看看，这半年收获与得失，为下一个季度做一点规划。
---

2019 年一转眼就过完了一半。正好 6 月的最后两天是周末，可以有时间回顾一下这半年来的生活和工作情况。本来纯脑回忆的时候，感觉这
半年什么都没有做就过完了。还好这两年形成的每日记录的习惯，让我可以更详细的回忆起这半年的生活：

* 1 月：CTP 1.0 版
* 2 月：使用 Julia 写传统量化策略；《如何阅读一本书》；BERT 初试；
* 3 月：Julia 做策略因子搜索；vscode-wordcount-cjk 问题修复；BERT 做多分类；
* 4 月：优化 Julia 策略搜索；商品期货的深度学习模型；统计学的课程
* 5 月：MediaV 新版；pyctpclient；MediaV 文章生成；DailyTick 创建
* 6 月：MediaV 新版；

粗略的列下来，好像还是有一点收获的。但再仔细盘点一下，又觉得收获非常有限。

### 1. CTP 交易服务器的 1.0 版

这个项目大概写了两个月，从 2018 年底（年会之前）就开始搞，一直到 2019 年 1 月才搞定大部分基础功能。之所以这么麻烦，是因为使用了太多的组件。
比如使用 docker 做的多进程管理；使用 ELK 做的监控；使用 RabbitMQ 做的消息中间件；使用 nuxt 写的 Web UI。现在看来，这些努力都没什么用处。
比如基于 docker 的多里程架构，其实不一定比多线程的架构要好。因为多进程了，所以通信起来要麻烦得多，控制的时候也麻烦。加上中间多了一个 RabbitMQ，
全异步的架构，让开发难度上升了好几个量级，而实际上确只解决了非常有限的问题。ELK 日志监控也是一样的，根本没人去看。timescaledb 因为和大家的习惯
不同，虽然性能更好，更适合做日间序列的存储，但谁在乎呢？最后不还是在使用 MySQL。使用 nuxt 写的 Web UI，虽然代码简单了，但因为没有 ORM 和 Web
的基础功能，扩展起来其实其实非常不方便。

后来，在商品期货被引入之后，又因为 24 小时运行导致 trader 的状态发生了奇怪的错位。而且商品期货的交易模式也和股指不一样，这又引入了新的复杂性。
总的来说，这一次的项目经验就是：不要写太复杂的东西！在思维架构的时候，要先老柴简单的方案。

当然，这个项目虽然选型不是很成功，但也尝试了一下微服务架构，另外对 CTP 的接口也有了很多了解，这棕后面花了更少的时间，实现了 pyctpclient。

### 2. Julia 策略搜索

在完成了 CTP 之后，本来是要投入精力开始搞模型的，不过，又花了一些时间（大概两个月）去搞了一个传统量化因子搜索项目。正好年初的时候想使用 Julia
来实现一些机器学习算法。于是，就借助这个项目，学习/练习了一下 Julia 的使用。虽然 Julia 用起来很爽，但这个项目本身的结果并不尽如人意。在完成了
基础的一些搜索工作，和一些基本优化以后，这个项目就停止了。在这个项目中，最大的收获当然是对 Julia 的使用和熟悉，这也使我更加相信 Julia 会成为
取代 Python 的下一个数值计算/机器学习语言。

### 3. MediaV 新版开发

MediaV 是我加入这家公司以后，接手的第一个项目，后来这个项目的合同又重新签定，要完成一个新的版本，做为对外的 SaaS 服务。所以就需要对原来的模型
进行重新设计。其实这个项目可以实践的方面还挺多的，除了现在的 django 开发，以及从一开始就实行的前后端分离，到背后的 NLP 任务都很值得作为一个
非常有前景的项目去开发。但公司现在主要的目标是金融，所以，想要深入到 NLP 的更多工作中的话，我需要先完成一个可以使用的金融模型才行。说到 NLP，其实
在 2 月和 3 月的时候，是尝试过使用 BERT 来做一些分类的。但最后还是因为数据量太少而无法完成。现在 XLNet 已经出来了，虽然从最近的了解中，感觉这东西
在文本不长的时候，不会比 BERT 好多少，但 NLP 的新时代已经开始了。我在想，怎么使用 Julia 来实现一波 NLP 的模型和算法，这样就可以一边推进 Julia
更快的蚕食 Python 的市场，一边又能进入 NLP 的神奇世界了。

在 6 月底，这个项目的主体工作已经完成，但多少还有一些问题需要修。

### 4. 阅读和学习

这半年来，想学的和尝试学的东西有很多，从一开始的 《Pattern Recognition and Machine Learning》、《机器学习》到后来的概率论、概率与统计，中间还
有《Software Foundations》。前前后后，从书到视频到听音频，折腾了好几次。但现在回顾的时候，感觉什么也没学到，这真是一个大问题！如果我看了一堆东西，
最后什么也没记住，那我的学习还有什么意义呢？这是下半年需要着力改进的地方。

### 5. 放弃

#### 1. CTP C API

这个本来是为 Julia 准备的，不过后来因为 pyctpclient 的开发，我觉得那套 API 更好用一点。另外公司后面可能会使用云平台进行训练和部署，
所以 Julia 只能是我自己玩一下了。

### 2. CTP nng 版

因为 pyctpclient 的开发，这种复杂的服务器版就不再需要了。

## 今年计划了，但要被放弃的事

回看了一下年初的计划，很多年初以为可以完成的，现在要标记为放弃了。

### 1. Julia Machine Learning Cookbook

这个本来是今年最大的目标，但真动手做起来，发现自己现在掌握的知识还远不够写一本书的。所以先学习，做一些笔记、尝试和准备吧。

还有一些被放弃的：

1. 密码管理器
2. 机器学习的学习
3. 线性代数的学习
4. 放弃了《Pattern Recognition and Machine Learning》
5. 放弃了《机器学习》

很多事情被放弃，是因为现在不是做这个事情的时候，更多的原因是：精力真不够。要集中精力学习一些东西，不得不先放弃一睦和眼前任务无关的事情。

## 接下来三个月的关键目标

放弃了 Julia 的书之后，今年剩下的时间集中在两个目标上：

1. 概率与统计
2. 强化学习

额外的精力：

1. 批判性思维
2. NLP 模型的 Julia 实现

关于概率与统计学习，有三本书要阅读（也已经加入修订后的 2019 年计划中了）

1. 《有趣的统计：75招学会数据分析》
2. 《统计思维：程序员数学之概率统计》
3. 《Probability and Statistics》

强化学习目前没有对理论的深入学习的计划，只是通过 stable-baselines 的实践来进行学习。当然这样的结果是，对算法的理解不会太深入，但因为要实现强化学习
和算法难度还是很高的，所以不能一下就搞得很深入。

批判性思维就只有：《批判性思维工具：30天改变思维定势，学会独立思考(修订扩展版) 》这一本需要以实践的方式来阅读。至于 NLP 也是很从实践 BERT 开始，再
尝试去实现一些模型，比如 LSTM、Transformer 等。

鉴于上半年学习效果和效率不佳，下半年在学习的时候，以深入思考为主，强迫自己做笔记，并在笔记中思考。不可以直接抄书，只能总结和思考。这样，就算读得慢
一些，也要有足够的收获。毕竟要学，就一下子学透。因为后面再回来学习第二次、第三次的机会越来越少了。

OK，以上就是 2019 年上半年的总结和下半年的规划了。到 2019 年 9 月底的时候，会对下面的三个月的进展进行总结。希望那个时候，我会有足够多的收获吧。