"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np

import pandas as pd

from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset



class MEMOBenchDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. MEMO-Bench/dataset/)
        ann_path (string): Path to the CSV annotation file
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.ann_path = ann_path

        # 读取CSV文件并解析注释
        self.annotation = self._load_annotations()

    def _load_annotations(self):
        # 使用pandas加载CSV文件
        ann_df = pd.read_csv(self.ann_path)
        
        # 将CSV内容转化为字典列表
        annotations = []
        for idx, row in ann_df.iterrows():
            annotation = {
                "image": row["Image"],
                "score": row["Score"],
                "emotion": row["Distortions"]
            }
            annotations.append(annotation)
        
        return annotations

    def __getitem__(self, index):
        # 获取当前注释
        ann = self.annotation[index]

        # 构建图片路径并加载图像
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        # 进行视觉处理
        image = self.vis_processor(image)

        # 获取图像ID，这里假设ID是文件名中的最后一部分
        img_id = ann["image"]

        # 获取文本标签，这里将情感标签作为指令（例如：“Sad”, “Happy”）
        label = ann["emotion"]

        instruction_input = "[emotion] Please determine which emotion label in the image represents: neutral, sad, happy, angry, surprise, worried."
        # print(f"instruction_input: {instruction_input}")
        # print(f"self.text_processor(instruction_input): {self.text_processor(instruction_input)}")

        video_features = torch.zeros(1, 64, 1280)
        audio_features = torch.zeros(1, 64, 1024)

        # 返回数据字典
        return {
            "image": image,
            "image_id": img_id,
            "instruction_input": instruction_input,
            "video_features": video_features,
            "audio_features": audio_features,
            "answer": label,  # 此处是情感标签，作为模型的输入文本
            "score": ann["score"],  # 可选，评分可以作为附加信息返回
        }

    def __len__(self):
        return len(self.annotation)

