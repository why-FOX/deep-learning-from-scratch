# deep-learning-from-scratch
两个月深度学习上手ing
最近会更新的比较频繁

先推荐两门课（吴恩达的深度学习+李沐动手实践深度学习）

https://www.coursera.org/specializations/deep-learning

https://www.bilibili.com/video/BV1daQAYuEYm/?spm_id_from=333.1387.favlist.content.click

可以在这里快速过一遍神经网络基础原理

https://zhuanlan.zhihu.com/p/680222880

pytorch安装

还是推荐在官网previous version那里找到对应版本小于现在cuda版本的pytorch下载，比如我的cuda是12.3

CUDA 12.1

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

CNN架构代码更新（cnn_easy.py+MNIST）

先是基础版的卷积+全连接+池化层，用的是小数据集MNIST，环境已经给出

之后会有工程版或者是RESNET的代码更新~

先附上双语ResNet论文和讲解视频

https://ar5iv.labs.arxiv.org/html/1512.03385?_immersive_translate_auto_translate=1

https://www.bilibili.com/video/BV1P3411y7nn?spm_id_from=333.788.videopod.sections&vd_source=57b5eb2416b8f7dc96cadd2fbacbb622

反向梯度的核心：链式法则


关于全连接层比较好的博客

https://blog.csdn.net/weixin_45829462/article/details/106548749


