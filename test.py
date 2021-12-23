''' 测试 '''
from utils import read_data, preprocess, test, testing
from crf import CRF

# yidu-s4k
test_file = 'dataset/yidu-s4k/transformed_data/test_data.txt'
saved_file = 'model/yidu-s4k.pkl'

# 测试:基于实体粒度
# test(test_file, saved_file)

my_crf = CRF()
# # 预测
# my_crf.load(saved_file)
# test_sentence = '患者因“反复餐后腹胀、恶心1月余”于2015-07-03入住我院，外院未经提示胃癌，于2015-07-16在全麻上行胃癌姑息切除术（远端胃大切）'
# print('序列标注结果：')
# print('待测序列：', test_sentence)
# my_crf.predict(test_sentence)

# 提取实体
my_crf.load(saved_file)
test_sentence = '患者因“反复餐后腹胀、恶心1月余”于2015-07-03入住我院，外院未经提示胃癌，于2015-07-16在全麻上行胃癌姑息切除术（远端胃大切）'
print('待测序列：\n', test_sentence)
predicted = my_crf.predict(test_sentence)
print('预测标注：\n', predicted)
print('实体提取结果：')
my_crf.ner(test_sentence)