import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from minigpt4.datasets.datasets.mer_mme_datasets import *

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def save_to_jsonl(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='mer2023', help="dataset to evaluate")
args = parser.parse_args()
print(args)
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

os.makedirs(save_path, exist_ok=True)

print('============保存路径============')
print(save_path)
print('============保存路径============')

if 'SA' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["SA"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["SA"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["SA"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["SA"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_SA = os.path.join(save_path, "SA.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_SA, data_entry)

if 'ERLab' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["ERLab"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["ERLab"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["ERLab"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ERLab"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_ERLab = os.path.join(save_path, "ERLab.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_ERLab, data_entry)

if 'ERWild' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["ERWild"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["ERWild"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["ERWild"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["ERWild"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_ERWild = os.path.join(save_path, "ERWild.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_ERWild, data_entry)

if 'FGER' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["FGER"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["FGER"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["FGER"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["FGER"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_FGER = os.path.join(save_path, "FGER.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_FGER, data_entry)

if 'FGSA' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["FGSA"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["FGSA"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["FGSA"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["FGSA"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_FGSA = os.path.join(save_path, "FGSA.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_FGSA, data_entry)

if 'IR' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["IR"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["IR"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["IR"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["IR"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_IR = os.path.join(save_path, "IR.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_IR, data_entry)

if 'MLER' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["MLER"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["MLER"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["MLER"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["MLER"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_MLER = os.path.join(save_path, "MLER.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_MLER, data_entry)

if 'NoiseER' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["NoiseER"]["eval_file_path"]
    video_path = cfg.evaluation_datasets_cfg["NoiseER"]["video_path"]
    batch_size = cfg.evaluation_datasets_cfg["NoiseER"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["NoiseER"]["max_new_tokens"]

    data = SADataset(vis_processor, None, video_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    file_path_NoiseER = os.path.join(save_path, "NoiseER.jsonl")

    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        
        labels = batch['label']
        video_features = batch['video_features']
        audio_features = batch['audio_features']
        video_ids = batch['video_id']

        texts = prepare_texts(instruction_input, conv_temp)
        responses = model.generate(images, video_features, audio_features, instruction_input, max_new_tokens=max_new_tokens, do_sample=False)

        # 遍历 batch 内每个样本
        for i in range(len(video_ids)):
            data_entry = {
                "video_id": video_ids[i],
                "label": labels[i],
                "response": responses[i],
            }
            save_to_jsonl(file_path_NoiseER, data_entry)

print('============保存路径============')
print(save_path)
print('============保存路径============')

# conda activate emotion-llama_py310
# export CUDA_VISIBLE_DEVICES=5
# python eval_scripts/eval_mme_emotion_llama_v2.py --dataset
# # export PYTHONPATH=$PYTHONPATH:/home/user_name/big_space/project/Emotion-LLaMA-v2
# # torchrun --nproc_per_node 1 eval_scripts/eval_mer_emotion_llama_v2.py --cfg-path /home/user_name/big_space/project/Emotion-LLaMA-v2/eval_configs/emotionllamav2_mer_evaluation.yaml --dataset mer2023,meld
# # torchrun --nproc_per_node 1 eval_scripts/eval_mer_emotion_llama_v2.py --cfg-path /home/user_name/big_space/project/Emotion-LLaMA-v2/eval_configs/emotionllamav2_mer_evaluation.yaml --dataset meld

