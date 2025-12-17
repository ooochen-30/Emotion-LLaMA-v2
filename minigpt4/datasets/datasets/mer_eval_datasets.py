import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json
from copy import deepcopy

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2


class MEREvalDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        """
        通用多模态情绪识别数据集父类
        Args:
            vis_processor: 图像或视频处理器（如 transforms）
            text_processor: 文本预处理函数或Tokenizer
            vis_root: 图像或视频文件的根路径
            ann_path: 注释文件路径（支持CSV/JSON/JSONL）
        """
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        self.dataset_root = os.path.dirname(vis_root)
        self.feature_root = os.path.join(self.dataset_root, "features")

        self.annotation = self._load_annotations(ann_path)
        peak_path = ann_path.rsplit('_', 1)[0] + '_peak.json'
        
        with open(peak_path, 'r') as f:
            self.peak_info = json.load(f)

        if 'video_id' not in self.annotation.columns:
            raise ValueError("Annotation must contain 'video_id' column")

        self.audio_instruction_pool = [
            "Based on the audio input, what can you infer about the speaker's emotional state? Please consider tone, intonation, pitch, and speech rate in your analysis.",
        ]
        self.video_instruction_pool = [
            "Based on the visual input, what can be inferred about the person's emotional state? Please consider facial expressions, posture, and body language.",
        ]
        self.image_instruction_pool = [
            "What emotional state might the person in the image be experiencing? Please support your answer using visual cues such as facial expression, body posture, and gaze direction.",
        ]


    def _load_annotations(self, ann_path):
        if ann_path.endswith('.csv'):
            return pd.read_csv(ann_path, dtype={"video_id": str})
        elif ann_path.endswith('.json'):
            return pd.read_json(ann_path)
        elif ann_path.endswith('.jsonl'):
            return pd.read_json(ann_path, lines=True)
        else:
            raise ValueError(f"Unsupported annotation format: {ann_path}")

    def _load_npy(self, video_id, model_name):
        feat_path = os.path.join(self.feature_root, model_name, video_id + '.npy')
        try:
            feat = np.load(feat_path, allow_pickle=True)

            if not np.issubdtype(feat.dtype, np.floating):
                print(f"⚠️ Converting non-float features in {video_id}: dtype={feat.dtype}")
                feat = feat.astype(np.float32)
            return torch.tensor(feat, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Failed to load {feat_path}: {e}")
            if 'EVA-ViT-G' in model_name:
                return torch.zeros(1, 64, 1408, dtype=torch.float32)
            else:
                return torch.zeros(1, 64, 1280, dtype=torch.float32) # fallback placeholder

    def _extract_frame(self, video_path, max_retries=5):
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            raise ValueError(f"❌ Cannot open video file: {video_path}")

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"❌ Video has no frames: {video_path}")

        filename = os.path.basename(video_path)   
        name_only = os.path.splitext(filename)[0]
        peak_idx = self.peak_info[name_only]['peak_index']
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, peak_idx)
        success, frame = video_capture.read()

        if not success or frame is None:
            print(f"⚠️ Failed to read peak frame, trying middle frame: {video_path}")
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            success, frame = video_capture.read()

            if not success or frame is None:
                print(f"⚠️ Failed to read middle frame, trying first frame: {video_path}")
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = video_capture.read()
        video_capture.release()

        if not success or frame is None:
            raise ValueError(f"❌ Failed to read any frame from video: {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in subclasses.")


class MER2023EvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: angry, happy, neutral, sad, surprise, worried. Please ensure that all predicted labels come only from this list."
        ]
        
        self.task_pool = ["emotion"]
        
        self.annotation = self.annotation[self.annotation['MER2023_mode'].isin(['test1', 'test2', 'test3'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        label = row.get("emotion_label", "unknown")
        text = row.get("chinese", "")
        mode = row.get("MER2023_mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")
        
        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool


        character_line = "The person in video says: {}. ".format(text)
        instruction = "<s>[INST] <Vid><VideoHere></Vid> {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }

        
class MER2024SEMIEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: angry, happy, neutral, sad, surprise, worried. Please ensure that all predicted labels come only from this list."

        ]
        self.task_pool = ["emotion"]
        
        self.annotation = self.annotation[self.annotation['MER2024_mode'].isin(['test_semi'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   

        label = row.get("emotion_label", "unknown")
        text = row.get("chinese", "")
        mode = row.get("MER2023_mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool

        character_line = "The person in video says: {}. ".format(text)
        instruction = "<s>[INST] <Vid><VideoHere></Vid> {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }

class MELDEmotionEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: anger, disgust, fear, joy, neutral, sad, surprise. Please ensure that all predicted labels come only from this list."
        ]

        self.task_pool = ["emotion"]

        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'dev'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   

        emotion_label = row.get("Emotion", "unknown")
        sentiment_label = row.get("Sentiment", "unknown")
        text = row.get("text", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool

        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }
        
    
class CAEREvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: anger, disgust, fear, happy, neutral, sad, surprise. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["emotion"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'validation'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   

        emotion_label = row.get("emotion_label", "unknown")
        text = row.get("text_whisper", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }
        
class DFEWEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: angry, disgust, fear, happy, neutral, sad, surprise. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["emotion"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'validation'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   

        emotion_label = row.get("label", "unknown")
        text = row.get("sentence", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }
        

class E3EvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: angry, disgusted, happy, sad, sarcastic, scared, shy, surprised. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["emotion"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'val'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        emotion_label = row.get("emotion", "unknown")
        text = row.get("conversation", "")
        mode = row.get("mode", "")
        person = row["person"]
        
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
            
        character_line = "{} in video says: {}. ".format(person, text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }
        
        
class IEMOCAPEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: Anger, Sadness, Happiness. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["emotion"]
        self.annotation = self.annotation[self.annotation['session'].isin(['Session5'])]
        target_emotions = ['anger', 'neutral', 'sadness', 'happiness', 'excited']
        self.annotation = self.annotation[self.annotation['official_label'].isin(target_emotions)]


    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        
        emotion_label = row.get("official_label", "unknown")
        if emotion_label == 'excited':
            emotion_label = 'happiness'
        text = row.get("text", "")
        mode = row.get("session", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool

        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }        
        
        
class MAFWEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["emotion"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        
        emotion_label = row.get("single_label", "unknown")
        text = row.get("text_whisper", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }        
       
class MAFWMultiEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_multi_label_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be one or more emotion labels strictly chosen from the following list: anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["multi-label"]
        self.annotation = self.annotation[
            (self.annotation['mode'].isin(['test'])) &
            (self.annotation['multi_label'].notna())
        ]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        multi_label = row.get("multi_label", "unknown")
        text = row.get("text_whisper", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "multi-label":
            answer = multi_label
            instruction_pool = self.emotion_multi_label_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }
        
class BOLDEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_multi_label_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be one or more emotion labels strictly chosen from the following list: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["multi-label"]
        self.annotation = self.annotation[
            (self.annotation['mode'].isin(['val']))
        ]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        multi_label = row.get("Emotion_label", "unknown")
        text = row.get("text_whisper", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "multi-label":
            answer = multi_label
            instruction_pool = self.emotion_multi_label_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }  
        
class MCEIUEmotionEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: anger, disgust, fear, happy, neutral, sad, surprise. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["emotion"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'validation'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        emotion_label = row.get("emotion", "unknown")
        text = row.get("Subtitle", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }        
        
        
class MCEIUIntentEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.intent_instruction_pool = [
            "You are a helpful assistant specialized in **multimodal intent recognition**. Given the input video and audio, your task is to identify the **speaker's communicative intent** based on both visual and auditory information. Your output must be **only one intent label**, strictly chosen from the following list: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["intent"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'validation'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        intent_label = row.get("intent", "unknown")
        text = row.get("Subtitle", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "intent":
            answer = intent_label
            instruction_pool = self.intent_instruction_pool
        

        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }                
        
        
        
class MELDSentimentEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.sentiment_instruction_pool = [
            "You are a helpful assistant specialized in multimodal sentiment analysis. Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. Your output must be only one sentiment label strictly chosen from the following list: Neutral, Negative, Positive. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["sentiment"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'dev'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        emotion_label = row.get("Emotion", "unknown")
        sentiment_label = row.get("Sentiment", "unknown")
        text = row.get("text", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = sentiment_label
            instruction_pool = self.sentiment_instruction_pool

        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }        
        
        
class CMUMOSIEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.sentiment_instruction_pool = [
            "You are a helpful assistant specialized in multimodal sentiment analysis. Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. Your output must be only one sentiment label strictly chosen from the following list: Neutral, Negative, Positive. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["sentiment"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'valid'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        sentiment_label = row.get("sentiment_label", "unknown")
        text = row.get("text", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = sentiment_label
            instruction_pool = self.sentiment_instruction_pool
            
        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }          
        
   
class CMUMOSEIEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.sentiment_instruction_pool = [
            "You are a helpful assistant specialized in multimodal sentiment analysis. Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. Your output must be only one sentiment label strictly chosen from the following list: Neutral, Negative, Positive. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["sentiment"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test', 'valid'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row['video_id']+"_"+str(row['clip_id'])
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        sentiment_label = row.get("sentiment_label", "unknown")
        text = row.get("text", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = sentiment_label
            instruction_pool = self.sentiment_instruction_pool

        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }
   
   
class CHSIMSV2SEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.sentiment_instruction_pool = [
            "You are a helpful assistant specialized in multimodal sentiment analysis. Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. Your output must be only one sentiment label strictly chosen from the following list: Neutral, Negative, Positive. Please ensure that all predicted labels come only from this list."
        ]
        self.task_pool = ["sentiment"]
        self.annotation = self.annotation[self.annotation['mode'].isin(['test'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        sentiment_label = row.get("annotation", "unknown")
        text = row.get("text", "")
        mode = row.get("mode", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = sentiment_label
            instruction_pool = self.sentiment_instruction_pool


        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }        
        

class SIMSEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.sentiment_instruction_pool = [
            "What sentiment is conveyed by the speaker when analyzing their vocal cues, facial movements, and textual content? Choose from: Positive, or Negative."
        ]
        self.task_pool = ["sentiment"]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        

        sentiment_label = row.get("val_label", "unknown")
        
        text = row.get("english", "unknown")
        mode = "test"
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = sentiment_label
            instruction_pool = self.sentiment_instruction_pool

        character_line = "The person in video says: {}. ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }      
      
        
class OVMERDEvalDataset(MEREvalDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)
        self.instruction_pool = [
            "Based on the multimodal content of the video (visuals, audio, and language), identify all emotions the person may be expressing. You may choose multiple labels. Your output can be any combination of valid emotion labels, such as: 'disgust,contempt', 'sadness,anxiety', 'anger,surprise', 'happiness,contempt', or 'anxiety,helplessness'. Use only labels from this list: angry, happy, joyful, dissatisfied, worry, positive, excited, surprised, concern, relaxed, sad, serious, nervous, pleasant, accusation, concerned, warning, confused, delighted.",
        ]
        
        self.task_pool = ["multi-label"]

        

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = row["video_id"]
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   
        
        label = "unknown"
        
        text = row.get("sentence", "unknown")
        mode = "test"
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        
        answer = label
        instruction_pool = self.instruction_pool
            
        character_line = "The person in video says: {} ".format(text)
        
        instruction = "<s>[INST] <Vid><VideoHere></Vid>  {} [{}] {}  [/INST]".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
            "mode": mode,
        }      