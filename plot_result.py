''' 绘制k折交叉验证结果'''
import matplotlib.pyplot as plt
import json
import numpy as np

# # 设置汉字格式
# import pylab as mpl  # import matplotlib as mpl
# # 设置汉字格式
# mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体

with open('model/test_results.json', 'r') as f:
    test_results = json.load(f)

k = 3
Losses = []
Precisions = []
Recalls = []
F1s = []
for i in range(k):
    # print(test_results[i])
    test_result = test_results[i]
    Losses.append(np.array(test_result['Loss'][:20]))
    Precisions.append(np.array(test_result['Precision'][:20]))
    Recalls.append(np.array(test_result['Recall'][:20]))
    F1s.append(np.array(test_result['F1-Score'][:20]))

Losses = np.asarray(Losses)
Precisions = np.asarray(Precisions)
Recalls = np.asarray(Recalls)
F1s = np.asarray(F1s)

Loss_mean = np.mean(Losses, axis=0)
Loss_std = np.std(Losses, axis=0)
print(np.std(Losses))
Precision_mean = np.mean(Precisions, axis=0)
Precision_std = np.std(Precisions, axis=0)
print(Precision_mean)
Recall_mean = np.mean(Recalls, axis=0)
Recall_std = np.std(Recalls, axis=0)
print(Recall_std)
F1_mean = np.mean(F1s, axis=0)
F1_std = np.std(F1s, axis=0)
print(F1_std)

sizes = range(len(Loss_mean))
fig = plt.figure()
# plt.title('CRF')
ax1 = fig.add_subplot(111)
ax1.plot(sizes, Loss_mean, 'o-', color='r', label='loss')
ax1.fill_between(sizes, Loss_mean-Loss_std, Loss_mean+Loss_std, alpha=0.15, color='r')
ax1.set_ylabel('Loss', fontsize=13)
ax1.set_xlabel('Epoch', fontsize=13)
ax2 = ax1.twinx()
ax2.plot(sizes, Precision_mean, 'o-', color='gold', label='Precision')
ax2.fill_between(sizes, Precision_mean-Precision_std, Precision_mean+Precision_std, alpha=0.15, color='gold')
ax2.plot(sizes, Recall_mean, 'o-', color='olive', label='Recall')
ax2.fill_between(sizes, Recall_mean-Recall_std, Recall_mean+Recall_std, alpha=0.15, color='olive')
ax2.plot(sizes, F1_mean, 'o-', color='purple', label='F1-Score')
ax2.fill_between(sizes, F1_mean-F1_std, F1_mean+F1_std, alpha=0.15, color='purple')
ax2.set_ylabel('Evaluating Indicator', fontsize=13)
ax1.legend(loc='upper left')
ax2.legend(loc='center right')
plt.xticks(range(0,20,2))
plt.show()