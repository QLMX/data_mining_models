# 数据挖掘类比赛常用模型

## 1. 概述

该工程主要是几个比较常用的模型，包含二分类模型、多分类模型以及回归模型。他们分别是基于lightgbm实现、xgboost实现和keras实现：

* lightgbm
  * [binary_class.py](./code/lgb/binary_class.py):lightgbm实现的二分类
  * [multi_class.py](./code/lgb/multi_class.py):lightgbm实现的多分类
  * [regression.py](./code/lgb/regression.py):lightgbm实现的回归
  * [multi_class_custom_feval.py](./code/lgb/multi_class_custom_feval.py):lightgbm 自定义评价函数实现多分类
  * [multi_class_weight_loss.py](./code/lgb/multi_class_weight_loss.py):lightgbm多类别不平衡问题，实现类别加权优化
* xgboost
  * [binary_class.py](./code/xgb/binary_class.py):xgboost实现的二分类
  * [multi_class.py](./code/xgb/multi_class.py):xgboost实现的多分类
  * [regression.py](./code/xgb/regression.py):xgboost实现的回归
* keras实现的mlp
  * [binary_class.py](./code/keras/binary_class.py):keras实现的mlp,做二分类任务
  * [multi_class.py](./code/keras/multi_class.py):keras实现的mlp,做多分类任务
  * [regression.py](./code/keras/regression.py):keras实现的mlp,做回归任务
* pytorch实现的mlp
  * [binary_class.py](./code/pytorch/binary_class.py):pytorch实现的mlp,做二分类任务
  * [multi_class.py](./code/pytorch/multi_class.py):pytorch实现的mlp,做多分类任务
  * [regression.py](./code/pytorch/regression.py):实现的mlp,做回归任务

## 2.环境设置

可以直接通过`pip install -r requirements.txt`安装指定的函数包，具体的函数包如下：

```python
pandas
numpy
matplotlib
sklearn
tensorflow==1.12.0
keras==2.2.4
pytorch
seaborn
lightgbm==2.2.1
xgboost==0.90
```

## 3. 代码讲解

* [基于lightgbm实现的二分类、多分类和回归任务](https://mp.weixin.qq.com/s/t6EpWmLWP81DcJ7AUro3Ng)
* [基于xgboost实现的二分类、多分类和回归任务](https://mp.weixin.qq.com/s/Td0Vrx9YO5rEn66L4C42Zw)
* [基于keras实现的二分类、多分类和回归任务](https://mp.weixin.qq.com/s/XaB1BsLL_Va7dGL0S0rUOQ)
* [基于pytorch实现的二分类、多分类和回归任务](https://zhuanlan.zhihu.com/p/80381974)

## 4. 最后

目前只整理了几个常用的模型，下一步会将用到的不错的代码同步进来，如果有问题或者发现有什么错误的地方需要交流可以通过微信公众号或者代码中的邮箱联系作者，也可以通过Issues提出问题。如果该项目对你有帮助，麻烦点击一下star，鼓励一下作者。如果有最新内容更新会第一时间在微信公众号内发布，希望能和你一起在AI方面前进，成长。

---

**小尾巴**

欢迎加群交流：

QQ群：AI成长社①：545702197

微信群：添加微信号：Derek_wen8，备注：加群

作者的知乎：[一休](https://www.zhihu.com/people/qlmx-61/activities), 知乎专栏：[ML与DL成长之路](https://zhuanlan.zhihu.com/c_1138029910563020800)

微信公号：AI成长社<img src="./result/wx.jpg" width = "80" height = "80" />



