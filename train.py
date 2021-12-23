''' 训练 '''
import math
import json
import numpy as np
from utils import read_data, preprocess, test
from crf import CRF


# yidu-s4k
# train_file = 'dataset/yidu-s4k/transformed_data/train_data.txt'
train_file = 'dataset/yidu-s4k/transformed_data/full_data.txt'
test_file = 'dataset/yidu-s4k/transformed_data/test_data.txt'
saved_file = 'model/yidu-s4k.pkl'


# # 训练：不含交叉验证
# word_lists, tag_lists = read_data(train_file)
# data = preprocess(word_lists, tag_lists)
# my_crf = CRF()
# # my_crf.train(data, word_lists, tag_lists)
# my_crf.train(data)
# my_crf.save(saved_file)
# 测试
# test(test_file, saved_file)


# # 训练：含交叉验证
# # 获取word与tag对应的集合
# word_lists, tag_lists = read_data(train_file)
# # 打乱数据
# shuffle = np.arange(len(word_lists))
# # np.random.shuffle(shuffle)
# print(shuffle)
# # print(type(word_lists))
# word_lists = np.asarray(word_lists)
# tag_lists = np.asarray(tag_lists)
# word_lists = word_lists[shuffle]
# tag_lists = tag_lists[shuffle]
# word_lists = word_lists.tolist()
# tag_lists = tag_lists.tolist()
#
# # 将训练集划分为K折(k=1时不含交叉验证)
# k = 3
# length = len(word_lists) // k  # 每折的长度
# k_fold_words = []
# k_fold_tags = []
# test_words = []
# test_tags = []
#
# for i in range(k):
#     start = i*length
#     end = (i+1)*length
#     test_words.append(word_lists[start:end])
#     test_tags.append(tag_lists[start:end])
#     k_fold_words.append(word_lists[:start] + word_lists[end:])
#     k_fold_tags.append(tag_lists[:start] + tag_lists[end:])
#
# test_results = []
# for i in range(k):
#     print('第%d折' % (i+1))
#     word_lists = k_fold_words[i]
#     tag_lists = k_fold_tags[i]
#
#     # 获取x2x：x向x映射的字典
#     data = preprocess(word_lists, tag_lists)
#
#     # 训练
#     my_crf = CRF()
#     # print(len(k_fold_words[4]))
#     test_result = my_crf.train(data, test_words[i], test_tags[i])
#     test_results.append(test_result)
#     my_crf.save(saved_file+str(i+1)+'.pkl')

# with open('model/crf/full/test_results.json', 'w') as f:
#     json.dump(test_results, f)

# for i in range(k):
#     # 测试
#     test(test_file, saved_file)





# 提取实体
# my_crf.load(saved_file)
# sentence = input('请输入一段话：')
# my_crf.ner(sentence)