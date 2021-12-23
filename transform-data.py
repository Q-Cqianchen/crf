''' 数据集预处理 '''

import json

# 1、txt转json
# lines = []
# with open('raw_data/train_data.json', encoding='utf8', mode='r') as f:
#     for line in f.readlines():
#         lines.append(line.rstrip()+',')
#
# with open('transformed_data/train_data.json', encoding='utf8', mode='w') as f:
#     for line in lines:
#         f.writelines(line+'\n')

# 2、每行一字对应一标签

# 疾病和诊断(SIC)-sickness：医学上定义的疾病和医生在临床工作中对病因、病生理、分型分期等所作的判断。
# 检查(DET)-detect：	影像检查（X线、CT、MR、PETCT等）+造影+超声+心电图，未避免检查操作与手术操作过多冲突，不包含此外其它的诊断性操作，如胃镜、肠镜等。
# 检验(ASS)-assay：	在实验室进行的物理或化学检查，本期特指临床工作中检验科进行的化验，不含免疫组化等广义实验室检查
# 手术(OPE)-operation：	医生在患者身体局部进行的切除、缝合等治疗，是外科的主要治疗方法。
# 药物(MED)-medical：	用于疾病治疗的具体化学物质。
# 解剖部位(ANA)-anatomy：	指疾病、症状和体征发生的人体解剖学部位。
with open('dataset/yidu-s4k/transformed_data/train_data.json', encoding='utf8', mode='r') as f:
    words_list = json.load(f)
entities = words_list[0]['entities']
for entity in entities:
    print(entity)

# lines = []
# for dic in words_list:  # 每一项都是字典
#     text = dic['originalText']
#     line = []
#     for word in text:
#         line.append(word+' O')
#     entities = dic['entities']
#
#     for entity in entities:
#         if entity['label_type'] == '疾病和诊断':
#             tag = 'SIC'
#         elif entity['label_type'] == '检查':
#             tag = 'DET'
#         elif entity['label_type'] == '检验':
#             tag = 'ASS'
#         elif entity['label_type'] == '手术':
#             tag = 'OPE'
#         elif entity['label_type'] == '药物':
#             tag = 'MED'
#         elif entity['label_type'] == '解剖部位':
#             tag = 'ANA'
#
#         # 如果是单字实体
#         if entity['end_pos'] - entity['start_pos'] == 1:
#             line[entity['start_pos']] = text[entity['start_pos']]+' S-'+tag
#         else:
#             line[entity['start_pos']] = text[entity['start_pos']]+' B-'+tag
#             for i in range(entity['start_pos']+1, entity['end_pos']-1):
#                 line[i] = text[i] +' M-'+tag
#             line[entity['end_pos']-1] = text[entity['end_pos']-1]+' E-'+tag
#         # for word in text[entity['start_pos']:entity['end_pos']]:
#     lines.append(line)
#
# with open('transformed_data/train_data.txt', encoding='utf8', mode='w') as f:
#     for line in lines:
#         for word in line:
#             f.writelines(word+'\n')
#         f.writelines('\n')


# 测试集转换
# 1、txt转json
# lines = []
# with open('raw_data/test_data.json', encoding='utf8', mode='r') as f:
#     for line in f.readlines():
#         lines.append(line.rstrip()+',')
#
# with open('transformed_data/test_data.json', encoding='utf8', mode='w') as f:
#     for line in lines:
#         f.writelines(line+'\n')

# 2、每行一字对应一标签

# 疾病和诊断(SIC)-sickness：医学上定义的疾病和医生在临床工作中对病因、病生理、分型分期等所作的判断。
# 检查(DET)-detect：	影像检查（X线、CT、MR、PETCT等）+造影+超声+心电图，未避免检查操作与手术操作过多冲突，不包含此外其它的诊断性操作，如胃镜、肠镜等。
# 检验(ASS)-assay：	在实验室进行的物理或化学检查，本期特指临床工作中检验科进行的化验，不含免疫组化等广义实验室检查
# 手术(OPE)-operation：	医生在患者身体局部进行的切除、缝合等治疗，是外科的主要治疗方法。
# 药物(MED)-medical：	用于疾病治疗的具体化学物质。
# 解剖部位(ANA)-anatomy：	指疾病、症状和体征发生的人体解剖学部位。
with open('dataset/yidu-s4k/transformed_data/test_data.json', encoding='utf8', mode='r') as f:
    words_list = json.load(f)
print(words_list[0])

# lines = []
# for dic in words_list:  # 每一项都是字典
#     text = dic['originalText']
#     line = []
#     for word in text:
#         line.append(word+' O')
#     entities = dic['entities']
#
#     for entity in entities:
#         if entity['label_type'] == '疾病和诊断':
#             tag = 'SIC'
#         elif entity['label_type'] == '影像检查':
#             tag = 'DET'
#         elif entity['label_type'] == '实验室检验':
#             tag = 'ASS'
#         elif entity['label_type'] == '手术':
#             tag = 'OPE'
#         elif entity['label_type'] == '药物':
#             tag = 'MED'
#         elif entity['label_type'] == '解剖部位':
#             tag = 'ANA'
#
#         # 如果是单字实体
#         if entity['end_pos'] - entity['start_pos'] == 1:
#             line[entity['start_pos']] = text[entity['start_pos']]+' S-'+tag
#         else:
#             line[entity['start_pos']] = text[entity['start_pos']]+' B-'+tag
#             for i in range(entity['start_pos']+1, entity['end_pos']-1):
#                 line[i] = text[i] + ' M-' + tag
#             line[entity['end_pos']-1] = text[entity['end_pos']-1]+' E-'+tag
#         # for word in text[entity['start_pos']:entity['end_pos']]:
#     lines.append(line)
#
# print(len(lines))
# with open('transformed_data/test_data.txt', encoding='utf8', mode='w') as f:
#     for line in lines:
#         for word in line:
#             f.writelines(word+'\n')
#         f.writelines('\n')