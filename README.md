# zhhyjs_baseline

数字中国创新大赛-智慧海洋建设 baseline

大赛链接：https://tianchi.aliyun.com/competition/entrance/231768/introduction

为了大家的公平，此代码丢弃了一些特征，大家可自行挖掘

有关围网等知识：

https://www.zhihu.com/question/22424944/answer/101042704?utm_source=wechat_session&utm_medium=social&utm_oi=819852327044386816

常规思路代码：

baseline.ipynb

路径识别代码：

分数较低，没有仔细调，只有0.35左右，给大家提供一个思路

FigureTransform.py:将路径保存为图片

resnet.py:模型文件

Dataset.py：数据读取文件

picture_baseline.py：模型训练及预测

不需要修改路径，可以直接运行，框架pytorch。
运行顺序： FigureTransform.py->picture_baseline.py
