import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


dic = {'SIC':0, 'DET':0, 'ASS':0, 'OPE':0, 'MED':0, 'ANA': 0}
with open('dataset/yidu-s4k/transformed_data/full_data.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        splited = line.split()
        if len(splited) == 2:
            word, tag = splited
            if tag != 'O':
                pos, entity = tag.split('-')
                if pos == 'B':
                    dic[entity] += 1

# 颜色
colors = ['black', 'k','dimgray', 'darkgrey', 'silver', 'gainsboro',]
colors1 = ['grey', 'gray', 'darkgrey', 'darkgray', 'silver', 'lightgray']
colors2 = ['salmon', 'tomato', 'darksalmon', 'lightsalmon', 'coral', 'orangered']
colors3 = ['olive', 'y', 'wheat', 'palegoldenrod', 'khaki', 'darkkhaki']
colors4 = ['lightslategray','powderblue']

label_list = ['疾病与诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位']
entities_num = [dic['SIC'], dic['DET'], dic['ASS'], dic['OPE'], dic['MED'], dic['ANA']]
# plt.hist()
print(dic)
plt.barh(range(6), entities_num, height=0.7, color=colors4, alpha=0.8)      # 从下往上画
plt.yticks(range(6), label_list, fontsize=12)
plt.title('实体类型分布情况')
plt.xlabel('个数', fontsize=12)
plt.show()
