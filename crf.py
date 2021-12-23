import numpy as np
import math
from scipy.optimize import fmin_l_bfgs_b
from ruamel import yaml
import pickle
import json

class CRF:
    def __init__(self):
        # 特征函数(字典的字典):先是关于模板的字典，其后是关于特征函数的字典
        self.feature_to_id = dict()
        self.feature_num = 0

        self.empirical_counts = None  # 非规范化经验分布
        self.params = None  # 各特征函数所对应的权重系数

        self.words_list = None
        self.tags_list = None
        self.word_to_id = None
        self.tag_to_id = None
        self.id_to_tag = None

        self.tag_num = None
        self.word_num = None

        self.featuresId_cluster = None  # 记录每一段话中，每一对(pre_tag,cur_tag)所包含的features_id
        self.gradient = None

        self.threshold = 1e150  # 指数上溢阈值
        self.sigma = 1e-10
        self.iter = 0  # 目前是第几个epoch
        self.loss = 0.0

        self.test_data = None
        self.tagged_test_data = None
        self.test_results = {'Loss':[], 'Precision':[], 'Recall':[], 'F1-Score':[]}

    def generate_templates(self, words, t):
        ''' 生成模板 '''
        length = len(words)
        templates = []
        templates.append('U[0]:%s' % words[t])
        if t > 0:
            templates.append('U[-1]:%s' % words[t-1])
            templates.append('B[-1,0]:%s%s' % (words[t-1], words[t]))
            if t > 1:
                templates.append('U[-2]:%s' % words[t-2])
                templates.append('B[-2,-1]:%s%s' % (words[t-2], words[t-1]))
        if t < length-1:
            templates.append('U[+1]:%s' % words[t+1])
            templates.append('B[0,+1]:%s%s' % (words[t], words[t+1]))
            if t < length-2:
                templates.append('U[+2]:%s' % words[t+2])
                templates.append('B[+1,+2]:%s%s' % (words[t+1], words[t+2]))
        return templates

    def generate_features(self):
        ''' 生成特征函数(pre_tag) '''
        empirical_counts = []
        for words, tags in zip(self.words_list, self.tags_list):
            # for word, tag in zip(words, tags):
            pre_tag = 0  # 前一个标签所对应的id，在一个句子开始时被初始化为0
            for t in range(len(words)):
                templates = self.generate_templates(words, t)
                cur_tag = self.tag_to_id[tags[t]]  # 目前标签所对应的id
                for template in templates:
                    # if template是标准的模板
                    if template in self.feature_to_id.keys():  # 如果已存在这个模板
                        # 转移特征函数
                        if (pre_tag, cur_tag) in self.feature_to_id[template].keys():
                            empirical_counts[self.feature_to_id[template][(pre_tag, cur_tag)]] += 1
                        else:
                            self.feature_to_id[template][(pre_tag, cur_tag)] = self.feature_num
                            # print(self.feature_num, template, (pre_tag, cur_tag))
                            empirical_counts.append(1.0)  # 追加第feature_num个元素，赋初值为1
                            self.feature_num += 1
                        # 状态特征函数
                        if cur_tag in self.feature_to_id[template].keys():
                            empirical_counts[self.feature_to_id[template][cur_tag]] += 1
                        else:
                            self.feature_to_id[template][cur_tag] = self.feature_num
                            # print(self.feature_num, template, cur_tag)
                            empirical_counts.append(1.0)  # 追加第feature_num个元素，赋初值为1
                            self.feature_num += 1
                    else:
                        self.feature_to_id[template] = dict()
                        # 转移特征函数
                        self.feature_to_id[template][(pre_tag, cur_tag)] = self.feature_num
                        # print(self.feature_num, template, (pre_tag, cur_tag))
                        empirical_counts.append(1.0)
                        self.feature_num += 1
                        # 状态特征函数
                        self.feature_to_id[template][cur_tag] = self.feature_num
                        # print(self.feature_num, template, (pre_tag, cur_tag))
                        empirical_counts.append(1.0)
                        self.feature_num += 1
                pre_tag = cur_tag  # 注意要更新

        # list 转 np.array
        self.empirical_counts = np.array(empirical_counts)

    def debug(self):
        ''' 目前用于验证是否出错 '''
        print(self.empirical_counts)
        print(self.feature_to_id)

    def get_potential_table(self, params, length, featuresId):
        ''' 计算一段话中各相邻节点所对应的势函数转移矩阵M '''
        # self.featuresId_cluster将训练集中各句子所包含的特征划分为一个个列表featuresId
        # featuresId将其中每个相邻对(pre_tag,cur_tag)所包含的特征划分为一个个列表features_id

        potential_tables = []
        for t in range(length):
            potential_table = np.zeros((self.tag_num, self.tag_num))
            for feature, feature_ids in featuresId[t].items():  # 获取该段话该字符处所响应特征函数的{feature,features_id}对
                score = sum(params[fid] for fid in feature_ids)
                if type(feature) == tuple:  # 如果是状态转移特征
                    # prev_y, y = feature
                    # potential_table[prev_y, y] += score
                    potential_table[feature] += score  # tuple可以直接进行索引
                else:  # 状态特征
                    potential_table[:, feature] += score
            potential_table = np.exp(potential_table)  # 指数势函数

            # 一段话的开始只能从开始符[s]转移到其他tag，而其他tag永不可能转到[s]，因此相应位置需置零
            if t == 0:
                potential_table[1:, :] = 0
            else:
                potential_table[:, 0] = 0
                potential_table[0, :] = 0
            potential_tables.append(potential_table)

        return potential_tables

    def forward_backward(self, length, potential_tables):
        ''' 前向后向算法 '''

        rec = [False] * length

        # alphas按行分开，alphas[t,:]为基于观测，该序列从[s]到第t步时处于各状态的非规范化概率
        alphas = np.zeros((length, self.tag_num))
        alphas[0, :] = potential_tables[0][0, :]

        for t in range(1, length):
            alphas[t, 1:] = np.matmul(alphas[t-1, :], potential_tables[t])[1:]
            if np.sum(alphas[t, :] > self.threshold) > 0:
                alphas[t-1, :] /= self.threshold
                alphas[t, 1:] = np.matmul(alphas[t-1, :], potential_tables[t])[1:]
                rec[t-1] = True

        # betas按行分开，betas[t,:]为基于观测，会经过第t步某状态处到达最后一步的非规范化概率
        betas = np.zeros((length, self.tag_num))
        betas[length-1, :] = 1.0
        for t in reversed(range(length-1)):
            betas[t, 1:] = np.matmul(potential_tables[t+1], betas[t+1,:])[1:]
            if rec[t]:
                betas[t] /= self.threshold

        z = sum(alphas[length - 1])  # 归一化因子
        return alphas, betas, z, rec

    def likelihood(self, params, *args):
        ''' 极大似然估计：注意，要极大化对数似然 L(params)，就是要极小化 -L(params) '''

        Z = 0  # 极大似然估计参数之一
        expectation = np.zeros(self.feature_num)
        for featuresId in self.featuresId_cluster:  # 逐段话
            length = len(featuresId)
            potential_tables = self.get_potential_table(params, length, featuresId)
            alphas, betas, z, rec = self.forward_backward(length, potential_tables)
            Z = Z + math.log(z) + sum(rec) * math.log(self.threshold)
            for t in range(length):  # 逐个字符
                # 获取该段话该字符处所响应特征函数的{feature,features_id}对
                for feature, feature_ids in featuresId[t].items():
                    if type(feature) == tuple:  # 如果是状态转移特征
                        pre_tag, cur_tag = feature
                        # # 如果一段话的开始不是[s]或中间字符的pre_tag是开始符
                        # if (t == 0 and pre_tag != 0) or (t != 0 and pre_tag == 0):
                        #     # print('wrong')
                        #     continue
                        if t == 0:
                            prob = (potential_tables[t][0, cur_tag] * betas[t, cur_tag]) / z
                        else:
                            prob = (alphas[t - 1, pre_tag] * potential_tables[t][pre_tag, cur_tag] * betas[t, cur_tag]) / z
                    else:
                        if rec[t]:
                            prob = (alphas[t, feature] * betas[t, feature] * self.threshold) / z
                        else:
                            prob = (alphas[t, feature] * betas[t, feature]) / z
                    for fid in feature_ids:
                        expectation[fid] += prob

        log_likelihood = np.dot(self.empirical_counts, params) - Z
        # print(log_likelihood, sum(abs(self.empirical_counts) > 1))

        # 加入了正则项的目标函数值
        loss = -log_likelihood + self.sigma/2 * np.sum(params**2)
        gradients = self.empirical_counts - expectation + self.sigma * params  # 加入正则项
        # gradients = self.empirical_counts - expectation
        self.gradient = gradients

        # return -log_likelihood
        self.loss = loss
        return loss

    def generate_featuresId_cluster(self, words_list, decoding=False):
        cluster = []
        for words in words_list:
            featuresId = []
            for t in range(len(words)):
                features_id = dict()
                templates = self.generate_templates(words, t)
                for template in templates:
                    if template not in self.feature_to_id.keys():  # 解码时可能存在该情况
                        continue
                    for feature, feature_id in self.feature_to_id[template].items():
                        # 过滤掉不符合要求的特征: 一段话的开始不是[s] or 中间字符的pre_tag是开始符
                        if type(feature) == tuple:
                            pre_tag, cur_tag = feature
                            if (t == 0 and pre_tag != 0) or (t != 0 and pre_tag == 0) :
                                continue

                        if feature in features_id.keys():
                            features_id[feature].add(feature_id)
                        else:
                            features_id[feature] = {feature_id}
                featuresId.append(features_id)
            cluster.append(featuresId)

        for featuresId in cluster:
            for features_id in featuresId:
                for feature, feature_id in features_id.items():
                    features_id[feature] = list(feature_id)  # 从set转为list

        if decoding:  # 如果是在解码
            return cluster
        else:
            self.featuresId_cluster = cluster

    def gra(self, params, *args):
        ''' 返回对数似然关于params的导数，注意取负号 '''
        return -self.gradient

    def train(self, data, test_data=None, tagged_test_data=None):
        # 验证集
        self.test_data = test_data
        self.tagged_test_data = tagged_test_data

        self.words_list = data['words_list']
        self.tags_list = data['tags_list']
        self.word_to_id = data['word_to_id']
        self.tag_to_id = data['tag_to_id']
        self.id_to_tag = data['id_to_tag']

        # 获取有多少个种类的标签和字符
        self.tag_num = len(self.tag_to_id)
        self.word_num = len(self.word_to_id)

        print('正在生成特征函数...')
        self.generate_features()  # 生成特征函数
        print('特征数量：', self.feature_num)
        print('标签数量：', self.tag_num)
        print('词汇数量：', self.word_num)
        print('正在生成featuresId_cluster...')
        # 生成self.featuresId_cluster: 之所以传入两个self开头的参数是因为该函数会被其他功能重用
        self.generate_featuresId_cluster(self.words_list)

        # 开始训练
        print('开始训练')
        self.params = np.zeros(self.feature_num)
        self.params, log_likelihood, information = fmin_l_bfgs_b(func=self.likelihood, callback=self.callback,
                                                                 fprime=self.gra, x0=self.params, maxiter=25)
        # self.params, log_likelihood, information = fmin_l_bfgs_b(func=self.likelihood,
        #                                                          fprime=self.gra, x0=self.params, maxiter=20)

        # self.gd()
        print('保存模型')
        self.save()
        # print(self.params)
        # print(log_likelihood)
        # self.debug()
        return self.test_results

    def callback(self, params):
        self.params = params
        self.iter += 1
        # loss = self.likelihood(params)
        # loss = 0
        if self.test_data is None:
            print('epoch %d: Loss=%.2f' % (self.iter, self.loss))
        else:
            P, R, F1 = self.test()
            print('epoch %d: Loss=%.2f, Precision=%.3f, Recall=%.3f, F1-Score:%.3f' % (self.iter, self.loss, P, R, F1))
            self.test_results['Loss'].append(self.loss)
            self.test_results['Precision'].append(P)
            self.test_results['Recall'].append(R)
            self.test_results['F1-Score'].append(F1)

    def test(self):
        TP = 0  # True Positive
        len_p = 0  # TP+FP
        len_r = 0  # TP+FN
        # print(self.test_data[0])
        # print(self.tagged_test_data[0])
        # iter = 0
        for sentence, tagged_sentence in zip(self.test_data, self.tagged_test_data):
            predicted, predicted_entity_num = self.ner(sentence, testing=True)
            # if iter == 0:
            #     print(predicted)
            tp, true_entity_num = self.cal(tagged_sentence, predicted)
            TP += tp
            len_p += predicted_entity_num
            len_r += true_entity_num

            # iter += 1

        print('True Positive：', TP)
        if len_p == 0:
            P = 0
        else:
            P = TP / len_p  # 精确率
        R = TP / len_r  # 召回率
        # print('len_r:', len_r)
        if P+R == 0:
            F1 = 0
        else:
            F1 = 2*P*R / (P+R)
        # print('Precision：', P)
        # print('Recall：', R)
        # print('F1-Score：', F1)
        return P, R, F1

    def cal(self, tags, predicted):
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
                        suc = False  # 此时也不用break，让t滑动到标签结尾。因为predicted肯定错了
                    t += 1
                if suc:
                    tp += 1

        return tp, entity_num

    def decode(self, sentence):
        ''' 使用Viterbi算法解码 '''
        length = len(sentence)

        # rec[i,j]: 若确定第j+1个字符为状态i，则其前一个状态为rec[i,j]
        rec = np.zeros((self.tag_num, length))
        # 最优解码路径
        decoded = np.zeros(length, dtype=int)

        # 生成该段话需要用到的featuresId
        cluster = self.generate_featuresId_cluster([list(sentence)], decoding=True)[0]

        scores = np.zeros(self.tag_num).reshape(-1, 1)
        # for template in self.feature_to_id.keys():
        #     for feature, feature_id in self.feature_to_id[template].items():
        #         if type(feature) == tuple:
        #             pre_tag, cur_tag = feature
        #             if pre_tag == 0:
        #                 # print(1)
        #                 scores[cur_tag] += 1

        for t in range(length):
            tmp = np.zeros((self.tag_num, self.tag_num))
            tmp[:, 0] = -np.inf  # 在预测中不能出现[s]到其他tag的情况
            tmp[0, :] = -np.inf  # 在预测中不能出现其他tag到[s]的情况
            tmp += scores  # 在行上广播，首先获得上一轮的基础值

            for feature, features_id in cluster[t].items():
                if type(feature) == tuple:
                    tmp[feature] += sum(self.params[features_id])
                else:
                    tmp[:, feature] += sum(self.params[features_id])
            rec[:, t] = np.argmax(tmp, axis=0)
            scores = np.max(tmp, axis=0)
        decoded[length - 1] = np.argmax(scores)  # 最终值最大的作为终点tag

        # 回溯
        for t in reversed(range(length - 1)):
            decoded[t] = rec[decoded[t + 1], t + 1]
        return decoded

    def predict(self, sentence):
        decoded = self.decode(sentence)
        predicted = [self.id_to_tag[i] for i in decoded]
        # print(predicted)
        return predicted

    def get_entitynum(self, sentence):
        entity_num = -1
        decoded = self.decode(sentence)
        predicted = [self.id_to_tag[i] for i in decoded]
        for i in range(len(predicted)):
            if predicted[i] != 'O' and predicted[i] != '[s]':
                pos, tag = predicted[i].split('-')
                if pos == 'B' or pos == 'S':
                    entity_num += 1
        # print(predicted)
        return predicted, entity_num

    def ner(self, sentence, testing=False):  # 命名实体识别，找出实体
        predicted = self.predict(sentence)

        entity_types = dict()  # 有多少种实体类型
        for tag in self.id_to_tag.values():
            if tag != 'O' and tag != '[s]':
                entity_type = tag.split('-')[1]
                if entity_type not in entity_types.keys():
                    entity_types[entity_type] = []

        entity_type = ''
        entity = ''
        for i in range(len(sentence)):
            if predicted[i] != 'O' and predicted[i] != '[s]':
                pos, entity_type = predicted[i].split('-')
                if pos == 'E':
                    entity += sentence[i]
                    entity_types[entity_type].append(entity)
                    entity = ''
                elif pos == 'M':
                    entity += sentence[i]
                elif pos == 'B':
                    if len(entity) != 0:  # B转到B或M转到B
                        # entity_types[entity_type].append(entity)
                        entity = ''
                    entity += sentence[i]
                elif pos == 'S':  # 单字实体
                    entity_types[entity_type].append(sentence[i])
                    entity = ''

        if len(entity) != 0:  # 最后一个实体
            entity_types[entity_type].append(entity)

        if not testing:
            for entity_type, entities in entity_types.items():
                if len(entities) != 0:
                    print(entity_type, end='：')
                    for entity in entities:
                        print(entity, end=' ')
                    print()
        else:
            entity_num = 0
            for entity_type, entities in entity_types.items():
                entity_num += len(entities)
            return predicted, entity_num

    def save(self, filename):
        model = dict()
        # print(self.feature_to_id)
        model['feature_to_id'] = self.feature_to_id
        model['feature_num'] = self.feature_num
        model['id_to_tag'] = self.id_to_tag
        model['tag_num'] = self.tag_num
        model['params'] = list(self.params)
        with open(filename, 'wb') as f:
            # yaml.dump(model, f)
            pickle.dump(model, f)

    def load(self, filename):
        print('Loading...')
        with open(filename, 'rb') as f:
            # model = yaml.load(f, Loader=yaml.Loader)
            model = pickle.load(f)

        self.feature_to_id = model['feature_to_id']
        self.feature_num = model['feature_num']
        self.id_to_tag = model['id_to_tag']
        self.tag_num = model['tag_num']
        self.params = np.asarray(model['params'])
        # print('params', self.params)
        # print(type(self.params))