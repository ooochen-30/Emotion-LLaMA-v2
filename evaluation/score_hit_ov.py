import os
import re
import glob
import json
import scipy
import argparse
import numpy as np
import pandas as pd

EMOTION_WHEEL_ROOT = '/home/user/big_space/Emotion-LLaMA-v2/evaluation/emotion_wheel'

def jsonl_to_npz(jsonl_file_path, mode, npz_output_path):
    filenames = []
    fileitems = []

    target_mode = mode

    with open(jsonl_file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())
            if data.get("mode") == target_mode:
                filenames.append(data["video_id"])
                fileitems.append(f"[{data['response']}]")

    filenames = np.array(filenames)
    fileitems = np.array(fileitems)

    np.savez(npz_output_path, filenames=filenames, fileitems=fileitems)

def string_to_list(str):

    if type(str) == np.ndarray:
        str = str.tolist()

    if isinstance(str, list):
        return str
    
    if str == '':
        str = []
    elif pd.isna(str):
        str = []
    else:
        if str[0] == '[':  str = str[1:]
        if str[-1] == ']': str = str[:-1]
        str = [item.strip() for item in re.split('[\'\",]', str) if item.strip() not in ['', ',']]
    return str

def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value): value=""
            values.append(value)
    return values


#############################################
###### 从所有 emotion wheel 中读取情感词 ######
#############################################
# read xlsx and convert it into map format
def read_wheel_to_map(xlsx_path):
    store_map = {}
    level1, level2, level3 = "", "", ""

    df = pd.read_excel(xlsx_path)
    for _, row in df.iterrows():
        row_level1 = row['level1']
        row_level2 = row['level2']
        row_level3 = row['level3']

        # update level1, level2, level3
        if not pd.isna(row_level1):
           level1 = row_level1
        if not pd.isna(row_level2):
           level2 = row_level2
        if not pd.isna(row_level3):
           level3 = row_level3
        
        # store into store_map
        level1 = level1.lower().strip() 
        level2 = level2.lower().strip()
        level3 = level3.lower().strip()
        if level1 not in store_map:
           store_map[level1] = {}
        if level2 not in store_map[level1]:
           store_map[level1][level2] = []
        store_map[level1][level2].append(level3)
    return store_map

# 所有 emotion wheel 合并，可以生成 248 个候选单词 => 247 候选标签了，跟我修正 fearful 有关系
def convert_all_wheels_to_candidate_labels():
    candidate_labels = []
    for xlsx_path in glob.glob(EMOTION_WHEEL_ROOT + '/wheel*.xlsx'):
        # print (xlsx_path)
        store_map = read_wheel_to_map(xlsx_path)
        # print (store_map)
        for level1 in store_map:
           for level2 in store_map[level1]:
              level3 = store_map[level1][level2]
              candidate_labels.append(level1)
              candidate_labels.append(level2)
              candidate_labels.extend(level3)
    candidate_labels = list(set(candidate_labels))
    return candidate_labels
'''
Totally, we can generate 253 emotion-wheel labels
'''


###########################################
###### emotion wheel 上的 labels 扩增 ######
###########################################
# func: 合并两个 map 合并，map中的元素的list的
def func_merge_map(map1, map2):
    all_items = list(map1.keys()) + list(map2.keys())

    merge_map = {}
    for item in all_items:
        if item in map1 and item in map2:
            value = list(set(map1[item] + map2[item]))
        elif item not in map1 and item in map2:
            value = map2[item]
        elif item in map1 and item not in map2:
            value = map1[item]
        merge_map[item] = value
    return merge_map


# 1. 读取候选词的同义词形式
# label2wheel: 实现所有标签 -> emotion wheel 中的标签
def read_candidate_synonym_onerun(runname='run1'):

    ## read candidate labels
    wheel_labels = convert_all_wheels_to_candidate_labels()

    ## gain mapping
    label2wheel = {}
    synonym_path = os.path.join(EMOTION_WHEEL_ROOT, 'synonym.xlsx')
    df = pd.read_excel(synonym_path)
    for _, row in df.iterrows():

        # 建立 self-mapping
        raw = row[f'word_{runname}'].strip().lower()
        assert raw in wheel_labels, f'error in {raw}' # check openai returns
        if raw not in label2wheel:
            label2wheel[raw] = []
        label2wheel[raw].append(raw)

        # 建立 synonyms -> raw 映射
        synonyms = row[f'synonym_{runname}']
        synonyms = string_to_list(synonyms)
        for synonym in synonyms:
            synonym = synonym.strip().lower()
            if synonym not in label2wheel: 
                label2wheel[synonym] = []
            label2wheel[synonym].append(raw)
    return label2wheel


# 2. 调用 8 次 gpt-4o / gpt-3.5 生成同义词 [采用这种方式，让情感词覆盖的尽量完整]
# mapping_merge: 实现所有标签 -> emotion wheel 中的标签
def read_candidate_synonym_merge():
    mapping_run1 = read_candidate_synonym_onerun('run1')
    mapping_run2 = read_candidate_synonym_onerun('run2')
    mapping_run3 = read_candidate_synonym_onerun('run3')
    mapping_run4 = read_candidate_synonym_onerun('run4')
    mapping_run5 = read_candidate_synonym_onerun('run5') 
    mapping_run6 = read_candidate_synonym_onerun('run6') 
    mapping_run7 = read_candidate_synonym_onerun('run7') 
    mapping_run8 = read_candidate_synonym_onerun('run8') 
    mapping_merge = func_merge_map(mapping_run1,  mapping_run2)
    mapping_merge = func_merge_map(mapping_merge, mapping_run3)
    mapping_merge = func_merge_map(mapping_merge, mapping_run4)
    mapping_merge = func_merge_map(mapping_merge, mapping_run5)
    mapping_merge = func_merge_map(mapping_merge, mapping_run6)
    mapping_merge = func_merge_map(mapping_merge, mapping_run7)
    mapping_merge = func_merge_map(mapping_merge, mapping_run8)
    # print (f'label number: {len(mapping_merge)}') 
    return mapping_merge
'''
## 利用同义词，将 253 个 emotion wheel 单词扩充到 1255 个单词
'''

def read_format2raws():
    format2raws = {}

    format_path = os.path.join(EMOTION_WHEEL_ROOT, 'format.csv')
    raws = func_read_key_from_csv(format_path, 'name')
    formats = func_read_key_from_csv(format_path, 'format')
    for raw, format in zip(raws, formats):

        # 1. 建立 format 与 raw 之间的映射
        format = string_to_list(format)
        for format_item in format:
            if format_item not in format2raws:
                format2raws[format_item] = []
            format2raws[format_item].append(raw)
        
        # 2. 建立 raw 与 raw 之间的映射
        if raw not in format2raws:
            format2raws[raw] = []
        format2raws[raw].append(raw)
    # print (len(format2raws))
    return format2raws
'''
## 利用格式扩增，将 1255 个单词扩充到 7386 个单词
=> 采用上述两步操作，将 255 个单词，扩充到了 7386 个单词，增加了近40倍
'''


#########################################################################
######## 采用得到的mapping，去度量 gt and pred openset 之间的重叠度 ########
## => 所有评价指标，都是把不在词表中的元素直接剔除掉 [统一化处理，方便后续比较]
##########################################################################
def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + '/*.npy'):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason


# case1: 计算只依赖于 format_mapping 下的结果
def func_backward_case1(label, format_mapping, raw_mapping=None, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    return stage1_unique


# case2: 核心是保证 backward 过程中的唯一性
def func_backward_case2(label, format_mapping, raw_mapping, wheel_map=None):
    if label not in format_mapping:
        return ""
    
    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]
    
    stage2_labels = raw_mapping[stage1_unique]
    assert isinstance(stage2_labels, list)
    stage2_unique = sorted(stage2_labels)[0]
    return stage2_unique


def func_get_wheel_cluster(wheel='wheel1', level='level1'):
    # print (f'process on wheel: {wheel} level: {level}')
    xlsx_path = os.path.join(EMOTION_WHEEL_ROOT, f'{wheel}.xlsx')
    emotion_wheel = read_wheel_to_map(xlsx_path)
    # print (emotion_wheel.keys())
    wheel_map = {}

    # 1. 所有聚类到level1
    if level == 'level1':
        for level1 in emotion_wheel:
            wheel_map[level1] = level1
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level1
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level1

    # 2. 全部聚类到level2
    elif level == 'level2':
        for level1 in emotion_wheel:
            wheel_map[level1] = sorted(emotion_wheel[level1])[0] # level1 映射到一个固定的 level2 上
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level2
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level2
    return wheel_map

## 函数3：引入 emotion wheel 进行评价
def func_backward_case3(label, format_mapping, raw_mapping, wheel_map):
    if label not in format_mapping:
        return ""
    
    level1_whole = []
    for format in format_mapping[label]:
        for raw in raw_mapping[format]:
            level1_whole.append(raw)
    
    for level1 in sorted(level1_whole): # 保证了结果唯一性
        if level1 in wheel_map:
            return wheel_map[level1]
    return ""


# metric: ['case1', 'case2', 'case3'] 展示的是一种层级化的聚类结果，从而看出每层的必要性
def func_map_label_to_synonym(mlist, format_mapping, raw_mapping, wheel_map, metric='case1'):
    new_mlist = []
    for label in mlist:
        if metric.startswith('case1'): label = func_backward_case1(label, format_mapping)
        if metric.startswith('case2'): label = func_backward_case2(label, format_mapping, raw_mapping)
        if metric.startswith('case3'): label = func_backward_case3(label, format_mapping, raw_mapping, wheel_map)
        if label == '': continue # 如果找不到 backward 的词，就把他剔除
        new_mlist.append(label)
    return new_mlist


def calculate_openset_onehot_hitrate(name2gt=None, 
                                     openset_root=None, openset_npz=None, name2pred=None, 
                                     metric='case1', format_mapping=None, raw_mapping=None,
                                     inter_print=True):

    # read name2gt
    candidata_labels = list(set([name2gt[name] for name in name2gt]))
    if inter_print: print (f'candidata_labels: {candidata_labels}')
    
    # read name2pred
    if name2pred is None: # in
        if openset_root is not None:
            name2pred = func_get_name2reason(openset_root)
        elif openset_npz is not None: # in
            names = np.load(openset_npz, allow_pickle=True)['filenames']
            items = np.load(openset_npz, allow_pickle=True)['fileitems']
            name2pred = {}
            for (name, item) in zip(names, items):
                name2pred[name] = item

    # read all mapping 
    if format_mapping is None:
        format_mapping = read_format2raws()          # level3 -> level2
    if raw_mapping is None:
        raw_mapping = read_candidate_synonym_merge() # level2 -> level1
    if metric.startswith('case3'): # level1 -> cluster center
        _, wheelname, levelname = metric.split('_')
        wheel_map = func_get_wheel_cluster(wheelname, levelname)
    else:
        wheel_map = None

    # score calculation
    hitrates, mscores = [], []
    for name in name2gt:
       
        # (gt) process
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = list(set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric)))
        if len(gt) == 0: continue # we only calculate non-neutral label hit rate

        # (pred) process
        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = list(set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric)))
        
        # (candidata_labels) process
        candidates = list(set(func_map_label_to_synonym(candidata_labels, format_mapping, raw_mapping, wheel_map, metric)))
        # print(pred,gt)
        # metric calculation
        hitrates.append(len(set(pred) & set(gt)))
        if len(set(pred) & set(candidates)) == 0:
            mscores.append(0)
        else:
            score = len(set(pred) & set(gt)) / len(set(pred) & set(candidates))
            mscores.append(score)
    if inter_print: print (f'after filter sample number: {len(hitrates)}')
    avg_hitrate, avg_mscore = np.mean(hitrates), np.mean(mscores)
    if inter_print: print (f'avg_hitrate: {avg_hitrate}; avg_mscore: {avg_mscore}')
    return avg_hitrate, avg_mscore


def hitrate_metric_calculation(name2gt=None, openset_root=None, openset_npz=None, name2pred=None, inter_print=True):

    # 已 M-avg 为主指标 [全部映射到 level1 的 label]
    candidate_metrics = [
                        'case3_wheel1_level1',
                        'case3_wheel2_level1',
                        'case3_wheel3_level1',
                        'case3_wheel4_level1',
                        'case3_wheel5_level1',
                        ]

    # 计算每个metric的这个值
    whole_scores = []
    for metric in candidate_metrics:
        hitrate, mscore = calculate_openset_onehot_hitrate(name2gt=name2gt,
                                                           openset_root=openset_root, openset_npz=openset_npz, name2pred=name2pred,
                                                           metric=metric, format_mapping=format_mapping, raw_mapping=raw_mapping,
                                                           inter_print=inter_print)
        whole_scores.append([hitrate, mscore])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores


# 计算一个 （gt_root, openset_root）在 metric 下的重叠率
# => 修改代码：支持不同格式的输入文件，并且支持只计算样本子集的结果
# => 修改代码：支持外部读取 format_mapping 和 raw_mapping
def calculate_openset_overlap_rate(gt_root=None, gt_csv=None, name2gt=None, 
                                   openset_root=None, openset_npz=None, name2pred=None, 
                                   process_names=None, 
                                   metric='case1', format_mapping=None, raw_mapping=None,
                                   inter_print=True):

    # read name2gt
    if name2gt is None:
        if gt_root is not None:
            name2gt = func_get_name2reason(gt_root)
        elif gt_csv is not None:
            name2gt = {}
            names = func_read_key_from_csv(gt_csv, 'name')
            gts   = func_read_key_from_csv(gt_csv, 'openset')
            for (name, gt) in zip(names, gts):
                name2gt[name] = gt

    # read name2pred
    if name2pred is None:
        if openset_root is not None:
            name2pred = func_get_name2reason(openset_root)
        elif openset_npz is not None:
            names = np.load(openset_npz)['filenames']
            items = np.load(openset_npz)['fileitems']
            name2pred = {}
            for (name, item) in zip(names, items):
                name2pred[name] = item

    # process_names => (whole) / (subset)
    if process_names is None:
        process_names = [name for name in name2gt]

    # read all mapping 
    if format_mapping is None:
        format_mapping = read_format2raws()          # level3 -> level2
    if raw_mapping is None:
        raw_mapping = read_candidate_synonym_merge() # level2 -> level1
    if metric.startswith('case3'): # level1 -> cluster center
        _, wheelname, levelname = metric.split('_')
        wheel_map = func_get_wheel_cluster(wheelname, levelname)
    else:
        wheel_map = None

    # calculate (accuracy, recall) two values
    accuracy, recall = [], []
    for name in process_names:      
       
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric))

        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric))

        if len(gt) == 0: continue
        if len(pred) == 0:
            accuracy.append(0)
            recall.append(0)
        else:
            accuracy.append(len(gt & pred)/len(pred))
            recall.append(len(gt & pred)/len(gt))
    if inter_print: print ('process number (after filter): ', len(accuracy))
    avg_accuracy, avg_recall = np.mean(accuracy), np.mean(recall)
    if inter_print: print (f'avg acc: {avg_accuracy} avg recall: {avg_recall}')
    return  avg_accuracy, avg_recall


def wheel_metric_calculation(
    gt_root=None, 
    gt_csv='/home/user/big_space/Emotion-LLaMA-v2/evaluation/check-result-openset-chi-translate.csv',
    name2gt=None, 
    openset_root=None, 
    openset_npz=None, 
    name2pred=None,
    process_names=None, 
    inter_print=True, 
    level='level1'):

    candidate_metrics = [
                        'case3_wheel1_level1',
                        'case3_wheel2_level1',
                        'case3_wheel3_level1',
                        'case3_wheel4_level1',
                        'case3_wheel5_level1'
                        ]
    # 计算每个metric的这个值
    whole_scores = []
    for metric in candidate_metrics:
        precision, recall = calculate_openset_overlap_rate(gt_root=gt_root,
                                                        gt_csv=gt_csv,
                                                        name2gt=name2gt,
                                                        openset_root=openset_root, 
                                                        openset_npz=openset_npz, 
                                                        name2pred=name2pred,
                                                        process_names=process_names, 
                                                        metric=metric,
                                                        format_mapping=format_mapping,
                                                        raw_mapping=raw_mapping,
                                                        inter_print=inter_print)
        fscore = 2 * (precision * recall) / (precision + recall)
        whole_scores.append(fscore)
        # whole_scores.append([fscore, precision, recall])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    # return whole_scores
    return whole_scores,avg_scores

def calculate_ov_zeroshot(openset_npz, inter_print=True):
    # 计算 EW-based metrics
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    whole_fscores,avg_scores = wheel_metric_calculation(name2gt=None, name2pred=name2pred, inter_print=inter_print)
    return whole_fscores,avg_scores


if __name__ == "__main__":
    format_mapping = read_format2raws()   
    raw_mapping = read_candidate_synonym_merge() # level2 -> level1

    hitrate_gt_directory = '/home/user/big_space/Emotion-LLaMA-v2/evaluation/hitrate_gt'
    datasets = ['mer2023','mer2024_semi','iemocap','meld_emotion','ov_merd+']
    
    parser = argparse.ArgumentParser(description="Evaluate hitrate and ov JSONL Results")
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    directory = args.path

    print(directory)
    for item in os.listdir(directory):
        if item.endswith('.npz'):
            continue
        if item.find('mer2023')!=-1:
            mode = 'test1'
            jsonl_path = os.path.join(directory,item)
            npz_path = jsonl_path.replace('jsonl','npz')
            if not os.path.exists(npz_path):
                jsonl_to_npz(jsonl_path,mode,npz_path)
            answer_npz = f'{hitrate_gt_directory}/gt_mer2023_test1.npz'
            data = np.load(answer_npz, allow_pickle=True)
            name2gt = {}
            for key in data.files:
                name2gt[key] = data[key].item() if hasattr(data[key], 'item') else data[key]
            hitrate, _ = hitrate_metric_calculation(name2gt=name2gt, openset_npz=npz_path, inter_print=False)
            print(f'mer2023:{hitrate}')

        if item.find('mer2024_semi')!=-1:
            mode = 'other'
            jsonl_path = os.path.join(directory,item)
            npz_path = jsonl_path.replace('jsonl','npz')
            if not os.path.exists(npz_path):
                jsonl_to_npz(jsonl_path,mode,npz_path)
            answer_npz = f'{hitrate_gt_directory}/gt_mer2024_semi_other.npz'
            data = np.load(answer_npz, allow_pickle=True)
            name2gt = {}
            for key in data.files:
                name2gt[key] = data[key].item() if hasattr(data[key], 'item') else data[key]
            hitrate, _ = hitrate_metric_calculation(name2gt=name2gt, openset_npz=npz_path, inter_print=False)
            print(f'mer2024_semi:{hitrate}')

        if item.find('iemocap')!=-1:
            mode = 'Session5'
            jsonl_path = os.path.join(directory,item)
            npz_path = jsonl_path.replace('jsonl','npz')
            if not os.path.exists(npz_path):
                jsonl_to_npz(jsonl_path,mode,npz_path)
            answer_npz = f'{hitrate_gt_directory}/gt_iemocap_Session5.npz'
            data = np.load(answer_npz, allow_pickle=True)
            name2gt = {}
            for key in data.files:
                name2gt[key] = data[key].item() if hasattr(data[key], 'item') else data[key]
            hitrate, _ = hitrate_metric_calculation(name2gt=name2gt, openset_npz=npz_path, inter_print=False)
            print(f'iemocap:{hitrate}')

        if item.find('meld_emotion')!=-1:
            mode = 'test'
            jsonl_path = os.path.join(directory,item)
            npz_path = jsonl_path.replace('jsonl','npz')
            if not os.path.exists(npz_path):
                jsonl_to_npz(jsonl_path,mode,npz_path)
            answer_npz = f'{hitrate_gt_directory}/gt_meld_emotion_test.npz'
            data = np.load(answer_npz, allow_pickle=True)
            name2gt = {}
            for key in data.files:
                name2gt[key] = data[key].item() if hasattr(data[key], 'item') else data[key]
            hitrate, _ = hitrate_metric_calculation(name2gt=name2gt, openset_npz=npz_path, inter_print=False)
            print(f'meld_emotion:{hitrate}')

        if item.find('ov_merd+')!=-1:
            model = 'test'
            jsonl_path = os.path.join(directory,item)
            npz_path = jsonl_path.replace('jsonl','npz')
            if not os.path.exists(npz_path):
                jsonl_to_npz(jsonl_path,model,npz_path)

            _, avg_scores = calculate_ov_zeroshot(npz_path, inter_print=False)
            print(f'ov_merd+:{avg_scores}')
            