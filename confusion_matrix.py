''' 绘制混淆矩阵'''
import matplotlib.pyplot as plt
import numpy as np
from utils import read_data, preprocess
from crf import CRF

test_file = 'dataset/yidu-s4k/transformed_data/test_data.txt'
word_lists, tag_lists = read_data(test_file)
data = preprocess(word_lists, tag_lists)
my_crf = CRF()
my_crf.load('model/crf/yidu-s4k.pkl')
# 初始化混淆矩阵
confusion = np.zeros((6,6), dtype='int')
et2id = {'SIC':0, 'DET':1, 'ASS':2, 'OPE':3, 'MED':4, 'ANA':5}
for words, tags in zip(word_lists, tag_lists):
    predicted = my_crf.predict(words)
    for p, t in zip(predicted, tags):
        if p == 'O' or t == 'O':
            continue
        else:
            pos_p, e_p = p.split('-')
            pos_t, e_t = t.split('-')
            confusion[et2id[e_p],et2id[e_t]] += 1

# 热度图
plt.imshow(confusion, cmap=plt.cm.OrRd)
# ticks 坐标轴的坐标点
indices = range(len(confusion))
plt.xticks(indices, ['SIC', 'DET', 'ASS', 'OPE', 'MED', 'ANA'])
plt.yticks(indices, ['SIC', 'DET', 'ASS', 'OPE', 'MED', 'ANA'])

plt.colorbar()

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("CRF's Confusion Matrix")

# plt.rcParams两行是用于解决标签不能显示汉字的问题
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 显示数据
for i in range(len(confusion)):    #第几行
    for j in range(len(confusion[i])):    #第几列
        # plt.text(i, j, confusion[i][j])
        plt.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.show()