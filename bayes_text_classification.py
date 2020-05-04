# 中文文本分类
import os,re
import jieba
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text=open(file_path, 'r', encoding='gb18030').read()
    # text = re.sub("[\s+\.\!\/_,$%^*(+\"\'\-\|\]\[\”★]+|[【】《》·；)`．：:+—！，。？?、~@#￥%…&*（）]+|[A-Za-z0-9]+", "", text)
    textcut = jieba.cut(text)
    text_with_spaces=' '.join(textcut)
    # print('结巴切割后的数据: ', text_with_spaces)
    return text_with_spaces

def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
        labels_list.append(label)                                                                                                                 
    return words_list, labels_list

def build_train_data():
    # 训练数据
    train_words_list1, train_labels1 = loadfile('./data/bayes_classfication_data/train/女性', '女性')
    train_words_list2, train_labels2 = loadfile('./data/bayes_classfication_data/train/体育', '体育')
    train_words_list3, train_labels3 = loadfile('./data/bayes_classfication_data/train/文学', '文学')
    train_words_list4, train_labels4 = loadfile('./data/bayes_classfication_data/train/校园', '校园')

    train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
    train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

    return train_words_list,train_labels

def build_test_data():
    # 测试数据
    test_words_list1, test_labels1 = loadfile('./data/bayes_classfication_data/test/女性', '女性')
    test_words_list2, test_labels2 = loadfile('./data/bayes_classfication_data/test/体育', '体育')
    test_words_list3, test_labels3 = loadfile('./data/bayes_classfication_data/test/文学', '文学')
    test_words_list4, test_labels4 = loadfile('./data/bayes_classfication_data/test/校园', '校园')

    test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
    test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4
    return test_words_list,test_labels



def train():
    #加载停用词表
    with open(r'./data/stopword.txt', 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    # 计算单词权重
    tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5)
    # 生成训练,测试用数据
    train_words_list, train_labels = build_train_data()
    train_features = tf.fit_transform(train_words_list)
    print('训练数据的形状: ', train_features.shape)
    print('不重复的词:', tf.get_feature_names())
    print('每个单词的 ID:', tf.vocabulary_)
    # 输出每个单词在每个文档中的 TF-IDF 值，向量里的顺序是按照词语的 id 顺序来的：
    print('每个单词的 tfidf 值:', train_features.toarray())

    # 生成朴素贝叶斯分类器
    clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

    # 模块5：使用生成的分类器做预测
    test_words_list, test_labels = build_test_data()
    test_features = tf.transform(test_words_list)
    print('测试数据的形状: ', test_features.shape)

    predicted_labels = clf.predict(test_features)
    # 计算准确率
    accuracy_score = metrics.accuracy_score(test_labels, predicted_labels)
    print('准确率为：', accuracy_score)


if __name__ == '__main__':
    train()

