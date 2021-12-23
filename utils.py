import numpy as np
from crf import CRF


def preprocess(words_list, tags_list):
    word_to_id = {}
    # id_to_word = {}
    tag_to_id = {'[s]': 0}  # 定义开始符号
    id_to_tag = {0: '[s]'}

    for words in words_list:
        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                # id_to_word[new_id] = word

    for tags in tags_list:
        for tag in tags:
            if tag not in tag_to_id:
                new_id = len(tag_to_id)
                tag_to_id[tag] = new_id
                id_to_tag[new_id] = tag

    # corpus = np.array([word_to_id[w] for w in words])
    data = dict()
    data['words_list'] = words_list
    data['tags_list'] = tags_list
    data['word_to_id'] = word_to_id
    data['tag_to_id'] = tag_to_id
    data['id_to_tag'] = id_to_tag
    # print(word_to_id)
    return data


def read_data(filename):
    words = []
    tags = []
    words_list = []
    tags_list = []
    with open(filename, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            if len(line) == 1:
                words_list.append(words)
                tags_list.append(tags)
                words = []
                tags = []
            else:
                splited = line.split()
                if len(splited) == 2:
                    word, tag = splited
                else:
                    word = ' '
                    tag = splited[0]

                # print(word,tag)
                words.append(word)
                tags.append(tag.rstrip())  # 去除尾部空行符

    return words_list, tags_list


def cal(tags, predicted):
    tp = 0
    t = 0
    length = len(tags)
    entity_num = 0
    while t < length:
        if tags[t] == 'O':  # 正确标注为O的不用管
            t += 1
        else:
            entity_num += 1
            suc = True
            while t < length and tags[t] != 'O':
                if tags[t] != predicted[t]:
                    # 如果predicted的最后一个的pos标签是M，仍认为它是正确的
                    # if tags[t][0] == 'E' and predicted[t][0] == 'M' and ((t+1 < length and predicted[t+1] == 'O') or t+1==length):
                    #     t += 1
                    #     continue
                    suc = False  # 此时也不用break，让t滑动到标签结尾。因为predicted肯定错了
                t += 1
            if suc:
                tp += 1
            # # 将标签pos-entity分解为pos和entity两部分
            # pos, entity = tags.split()
        if t == 1 or t == length-1:
            entity_num -= 1

    return tp, entity_num


def test(test_data_file, model_file):
    # 获取模型
    my_crf = CRF()
    my_crf.load(model_file)

    # 获取测试数据
    test_data = []
    tagged_test_data = []
    sentence = ''
    tagged_sentence = []
    with open(test_data_file, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            if line == '\n':
                test_data.append(sentence)
                tagged_test_data.append(tagged_sentence)
                sentence = ''
                tagged_sentence = []
            else:
                splited = line.split()
                if len(splited) == 2:
                    word, tag = splited
                else:
                    word = ' '
                    tag = splited[0]
                sentence += word
                tagged_sentence.append(tag)
    # print(test_data)
    # print(tagged_test_data)

    TP = 0  # True Positive
    len_p = 0  # TP+FP
    len_r = 0  # TP+FN
    for sentence, tagged_sentence in zip(test_data, tagged_test_data):
        # predicted, predicted_entity_num = my_crf.ner(sentence, testing=True)
        predicted, predicted_entity_num = my_crf.get_entitynum(sentence)
        tp, true_entity_num = cal(tagged_sentence, predicted)
        TP += tp
        len_p += predicted_entity_num
        len_r += true_entity_num
    print('True Positive：', TP)
    P = TP / len_p  # 精确率
    R = TP / len_r  # 召回率
    print('Precision：', P)
    print('Recall：', R)
    print('F1-Score：', 2*P*R/(P+R))


def testing(test_data_file, model_file):
    # 获取模型
    my_crf = CRF()
    my_crf.load(model_file)

    # 获取测试数据
    test_data = []
    tagged_test_data = []
    sentence = ''
    tagged_sentence = []
    with open(test_data_file, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            if line == '\n':
                test_data.append(sentence)
                tagged_test_data.append(tagged_sentence)
                sentence = ''
                tagged_sentence = []
            else:
                splited = line.split()
                if len(splited) == 2:
                    word, tag = splited
                else:
                    word = ' '
                    tag = splited[0]
                sentence += word
                tagged_sentence.append(tag)
    # print(test_data)
    # print(tagged_test_data)

    TP = 0  # True Positive
    len_p = 0  # TP+FP
    len_r = 0  # TP+FN
    for sentence, tagged_sentence in zip(test_data, tagged_test_data):
        predicted = my_crf.predict(sentence)
        for i in range(len(predicted)):
            if predicted[i] != 'O':
                len_p += 1
                if predicted[i] == tagged_sentence[i]:
                    TP += 1
            if tagged_sentence[i] != 'O':
                len_r += 1

    print('True Positive：', TP)
    P = TP / len_p  # 精确率
    R = TP / len_r  # 召回率
    print('Precision：', P)
    print('Recall：', R)
    print('F1-Score：', 2*P*R/(P+R))