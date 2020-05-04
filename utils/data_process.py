import os,re,jieba,random


def load_text():
    with open(file='../data/红楼梦/text.txt',mode='r',encoding='utf-8') as f:
        text_list = []
        for line in f.readlines():

            # 去掉中英文符号
            line_data = re.sub("[\s+\.\!\/_,$%^*(+\"\'\-\“]+|[`．：:”+——！，。？?、~@#￥%……&*（）]+|[A-Za-z0-9]+", "", line)
            line_data = line_data.strip()
            if len(line_data)>0:
                text_list.append(line_data)
        return text_list

def read_stop_words():
    data_list = []
    with open(r'../data/stopword.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip()
            data_list.append(data)
    return data_list

def fasttext_dataProcess(readFile,writeFile):
    """
    此API主要是将数据中的各种标点符号移除,包括中文的,英文的各种标点符号
    :param readFile:
    :param writeFile:
    :return:
    """
    with open(file=readFile,mode='r',encoding='utf-8') as f:
        data = f.read()
        data = ''.join(re.findall(r"([a-zA-Z0-9\u4E00-\u9FA5\s\r\n_]+)", data))
        with open(file=writeFile, mode='w', encoding='utf-8') as file_w:
            file_w.write(data)
    print('数据处理成功------')

def spilt_data(readFile,writePath, shuffle=False,ratio=0.2):
    """
    此API主要是将数据移除标点符号类的噪音数据,并且随机将数据按行切分为训练集和验证集
    :param readFile:
    :param shuffle:
    :param ratio:
    :return:
    """
    with open(file=readFile,mode='r',encoding='utf-8') as f:
        data = f.read()
        data = ''.join(re.findall(r"([a-zA-Z0-9\u4E00-\u9FA5\s\r\n_]+)", data))
        data_list=data.split(sep='\n')

        n_total = len(data_list)
        if ratio>0.5:
            ratio=1-ratio
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], data_list
        if shuffle:
            random.shuffle(data_list)
        data_tarin = '\n'.join(data_list[offset:])
        data_valid = '\n'.join(data_list[:offset])

        # 保存数据
        train_data=os.path.join(writePath, 'train.data')
        valid_data = os.path.join(writePath, 'valid.data')

        with open(file=train_data, mode='w', encoding='utf-8') as file_w:
            file_w.write(data_tarin)
        with open(file=valid_data, mode='w', encoding='utf-8') as file_w:
            file_w.write(data_valid)

        print('数据切割成功......')

if __name__ == '__main__':
    readFile='../data/fasttext_cook/cooking/cooking.txt'
    writePath='../data/fasttext_cook/cooking'

    spilt_data(readFile=readFile,writePath=writePath,shuffle=True,ratio=0.25)