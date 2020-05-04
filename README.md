# text_classfication_summary
## 这个项目是对对文本分类的一个总结

### >1使用贝叶斯分类器进行文本分类

* 代码在`bayes_text_classification.py`中
* 由于种种原因,我并没有对数据处理做的太多,目前的验证精度92%
* 我使用的是`MultinomialNB`分类器,即离散型朴素贝叶斯模型,或者叫多项式朴素贝叶斯模型

### >2使用fasttext对文本进行分类

* fasttext分类器的预测速度相当快,而且训练起来也比较方便
* 代码在`fasttext_classfication.py`中
* 如果你在windows下install使用fasttext,千万记得要首先安装C++的编译环境,然后在选择安装fasttext

* 如果你碰到在安装问题,也可联系我奥,adress:  nanyangjx@126.com



## 后话

* 文本分类也可基于深度学习模型,RNN/LSTM/GRU/BIGRU/BILSTM等,后续我会不断补充
* 敲一下重点,个人觉得,对于中文文本分类,反复处理好数据,才是王道
  * 比如,将数据,通过python正则,移除所有的标点符号
  * 其二,完善维护更加全面的停用词表,配合jieba使用
  * 维护更精准的自定义词典,配合结巴使用
