import numpy as np
import os,re
from fasttext import train_supervised,load_model


class fasttext_classfication(object):
    def __init__(self):
        pass

    def train_model(self, trainFilePath=None, dim=100, epoch=5, lr=0.1, loss='softmax',minCount=1):
        np.set_printoptions(suppress=True)
        model = f'./model/fastText_dataDim{str(dim)}_lr{str(lr)}_iter{str(epoch)}.model'

        if os.path.isfile(model):
            classifier = load_model(model)
        else:
            classifier = train_supervised(input=trainFilePath, label='__label__', dim=dim, epoch=epoch,
                                             lr=lr, wordNgrams=2, loss=loss,minCount=minCount)
            """
              训练一个监督模型, 返回一个模型对象
    
              @param input:           训练数据文件路径
              @param lr:              学习率
              @param dim:             向量维度
              @param ws:              cbow模型时使用
              @param epoch:           次数
              @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
              @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
              @param minn:            构造subword时最小char个数
              @param maxn:            构造subword时最大char个数
              @param neg:             负采样
              @param wordNgrams:      n-gram个数
              @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
              @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
              @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
              @param lrUpdateRate:    学习率更新
              @param t:               负采样阈值
              @param label:           类别前缀
              @param verbose:         ??
              @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
              @return model object
            """
            classifier.save_model(model)
        return classifier


    def cal_precision_and_recall(self,model_path='',testFilePath=''):
        if os.path.isfile(model_path):
            model = load_model(model_path)
            result = model.test(path=testFilePath)
            precision = result[1]
            recall = result[2]
            F1 = (2 * precision * recall) / (precision + recall)

            print('测试样本数据量 ', result[0])
            print('precision: {:.4f}'.format(precision))
            print('recall: {:.4f}'.format(recall))
            print('F1-precision: {:.4f}'.format(F1))
            return precision, recall, F1
        else:
            print('您提供了一个非法的模型路径......')

    def predict(self,model_path='',input="Why not put knives in the dishwasher?",k=2):

        input_ = ' '.join(re.findall(pattern=r"([a-zA-Z0-9\u4E00-\u9FA5]+)",string=input))

        if os.path.isfile(model_path):
            model = load_model(model_path)
            result = model.predict(input_, k)
            print(result)
            return result

if __name__ == '__main__':

    flag=2
    fs = fasttext_classfication()
    if flag==0:
        # 训练并保存模型
        train_path = f'./data/fasttext_cook/cooking.train'
        fs.train_model(trainFilePath=train_path,epoch=200)
    elif flag==1:
        # 计算模型准确率,召回率,F1
        _, _, _ = fs.cal_precision_and_recall(model_path='./model/fastText_dataDim100_lr0.1_iter200.model',
                                              testFilePath='./data/fasttext_cook/cooking.valid')
    elif flag == 2:
        #模型预测
        _ = fs.predict(model_path='./model/fastText_dataDim100_lr0.1_iter200.model',
                   input="Why not put knives in the dishwasher?")



