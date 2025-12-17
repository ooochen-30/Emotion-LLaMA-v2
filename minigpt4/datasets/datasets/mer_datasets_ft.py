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


class MERDatasetFT(Dataset):
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
        dataset = vis_root.split('/')[-2]
        peak_path = self.vis_root.replace('video', f'{dataset}_peak.json')
        
        with open(peak_path, 'r') as f:
            self.peak_info = json.load(f)

        if 'video_id' not in self.annotation.columns:
            raise ValueError("Annotation must contain 'video_id' column")
        
        self.multimodal_instruction_pool = [
            "Based on the video, including facial expressions, body language, vocal tone, and speech content, what can be inferred about the person's emotional state?",
            "What emotional impressions can be drawn from the person’s appearance, behavior, and voice in the video? Focus on facial expressions, gestures, tone of voice, and verbal cues.",
            "Analyze the person's emotional state by examining their facial expression, physical behavior, and how they speak. What multimodal signals contribute to your interpretation?",
            "Without relying solely on spoken content, what do the person’s facial expressions, body movements, and vocal qualities suggest about their emotional engagement?",
            "How would you describe the person's emotional tone by integrating visual behavior (e.g., facial tension, posture), vocal delivery, and the content of their speech?",
            "Observe the person in the video. What emotional cues emerge from their visual demeanor, voice inflection, and what they are saying?",
            "From a multimodal perspective, what signals—such as facial expressions, body posture, vocal pitch, and language use—reveal the person's emotional condition?",
            "What does the individual’s overall appearance, motion, voice characteristics, and speech content convey in terms of emotional state?",
            "How do the person's facial dynamics, gestures, intonation, and choice of words contribute to interpreting their emotional experience?",
            "What kind of mood is communicated through the combination of visual behavior, vocal tone, and speech? Consider congruence or discrepancy among these elements.",
            "What can be inferred about the person's emotions based on their facial expression, posture, tone of voice, and what they are verbally expressing?",
            "Describe observable emotional signals from the video. Pay attention to eye movement, vocal prosody, and emotionally loaded words.",
            "How do the person’s body language, voice tone, and verbal content collectively contribute to understanding their feelings at the moment?",
            "From a multimodal perspective, what emotional cues are conveyed by the person’s expression, behavior, vocal delivery, and speech meaning?",
            "Does the person appear emotionally engaged, tense, or indifferent based on their visual demeanor, vocal energy, and spoken words? Explain using concrete cues.",
            "If you had to guess the person’s emotional state from their expressions, voice, and language, what would it be? Justify your reasoning with observed indicators.",
            "How might hand positioning, head tilt, eye direction, vocal modulation, and specific phrases together indicate the person’s emotional condition?",
            "Are there subtle emotional cues—such as micro-expressions, slight tremors in the voice, or hesitant wording—that suggest underlying feelings?",
            "What does the visual and auditory scene suggest about how the person might be feeling? Consider their posture, voice strength, and emotional tone of speech.",
            "Without assumptions about the context, what visible gestures, vocal characteristics, or word choices might reflect the person’s internal state?",
            "What changes in the person's facial expression, posture, and voice occur throughout the video? Do they indicate any shift in emotional state?",
            "How does the person’s interaction with their environment—visually and verbally—reflect their current emotional disposition?",
            "What can be interpreted from the alignment between facial expression, bodily movement, voice tone, and linguistic content? Are they congruent or not?",
            "Does the person’s behavior and vocal delivery seem intentional or spontaneous? What might that reveal about their emotional state?",
            "Are there physical and vocal signs—like blinking rate, pitch breaks, or flinching—that point to emotional discomfort or surprise?",
            "What non-verbal and paralinguistic signals in the video (e.g., hesitations, sighs, avoidance gaze) could indicate inner conflict or uncertainty?",
            "Watch the video and describe the person’s emotional state by synthesizing facial expressions, vocal delivery, and the meaning of what is said.",
            "From the video clip, infer the person’s emotional state based on multimodal observations—what features support your conclusion?",
            "Please interpret the person's emotions by analyzing expressions, voice patterns, and verbal context. Which modality had the strongest impact on your judgment?",
            "Based on the video content, what emotion is most likely being expressed? Use visual, audio, and textual evidence to support your interpretation.",
            "What emotion do you think the person is experiencing? Justify your answer using cues from their body language, tone of voice, and speech content.",
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
            feat = np.load(feat_path)
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

    def _check_answer(self, answer):
        change_flag = False
        if not answer or not isinstance(answer, str):
            change_flag = True
        elif len(answer.strip().split()) > 200:
            change_flag = True
        return change_flag

    def _safe_str(self, val):
        import pandas as pd
        return "" if pd.isna(val) else str(val)
    
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

class MER2023DatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            "Identify the displayed emotion in the video: is it happy, sad, neutral, angry, worried, or surprise?",
            "Determine the emotional state shown in the video, choosing from happy, sad, neutral, angry, worried, or surprise.",
            "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, neutral, angry, worried, or surprise.",
            "Assess and label the emotion evident in the video: could it be happy, sad, neutral, angry, worried, surprise?",
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: angry, happy, neutral, sad, surprise, worried.",
            "Based on the video, determine which of the following emotion categories best describes the person's expression: sad, happy, angry, worried, neutral, or surprise.",
            "Please classify the observed emotional state in the video using one of the following: surprise, neutral, happy, sad, angry, or worried.",
            "Analyze the video and assign one of the following emotion labels to the person depicted: happy, worried, surprise, neutral, angry, or sad.",
            "After watching the video, decide which of the following emotions is being expressed: neutral, surprise, angry, sad, worried, or happy.",
        ]
        self.emotion_instruction_pool_no_neutral_class = [
            "Please determine which emotion label in the video represents: happy, sad, angry, worried, surprise.",
            "Identify the displayed emotion in the video: is it happy, sad, angry, worried, or surprise?",
            "Determine the emotional state shown in the video, choosing from happy, sad, angry, worried, or surprise.",
            "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, angry, worried, or surprise.",
            "Assess and label the emotion evident in the video: could it be happy, sad, angry, worried, surprise?",
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly chosen from the following list: angry, happy, neutral, sad, surprise, worried.",
            "Based on the video, determine which of the following emotion categories best describes the person's expression: sad, happy, angry, worried, or surprise.",
            "Please classify the observed emotional state in the video using one of the following: surprise, happy, sad, angry, or worried.",
            "Analyze the video and assign one of the following emotion labels to the person depicted: happy, worried, surprise, angry, or sad.",
            "After watching the video, decide which of the following emotions is being expressed: surprise, angry, sad, worried, or happy.",
        ]

        self.think_instruction_pool_no_neutral_class = [
            "You are given a video containing both visual and audio information. The possible emotion categories are: happy, angry, surprise, worried, sad. First, analyze the video by reasoning through facial expressions, gestures, tone of voice, and what is said. Write your reasoning inside the <think> and </think> tags. Then, select the most appropriate emotion and place it inside the <answer> and </answer> tags.",
            "Watch the video and consider all visual and auditory cues—facial expressions, body movements, voice pitch and tempo, and speech content. The emotion must be one of: happy, angry, surprise, worried, sad. Use <think> to explain your reasoning step by step, then put the final emotion label in <answer>.",
            "Analyze the multimodal signals in the video. The goal is to infer the person’s emotional state from the following options: happy, angry, surprise, worried, sad, disgust. In the <think> section, describe how visual and audio evidence supports your reasoning. Then write the most likely emotion label in <answer>.",
            "You need to determine which one of the following emotions is being expressed: happy, angry, surprise, worried, sad. Begin by reasoning through the video’s visual and auditory cues. Write this reasoning in <think>. Then, provide your chosen emotion in <answer>.",
            "Your task is to infer the emotion shown in the video using both what you see and hear. Possible categories: happy, angry, doubt, surprise, worried, sad. First, reason through the video in the <think> section. Include details such as facial expressions, body posture, tone, and wording. Then conclude with a final label in <answer>.",
            "Based on the speaker’s visual and auditory behavior in the video, choose one emotion from: happy, angry, surprise, worried, sad. In <think>, describe the visual cues (e.g., eye contact, gestures), the vocal tone (e.g., stressed, calm), and the meaning of what is said. Then, in <answer>, provide your final label.",
            "The video provides multimodal emotional cues. Consider the person’s expression, movement, tone, and speech. Possible emotion labels are: happy, angry, fear, surprise, worried, sad. Use <think> to perform step-by-step reasoning. Then, select the most suitable label and place it in <answer>.",
            "Watch and analyze the video content carefully. Take into account facial movements, body dynamics, vocal inflections, and linguistic content. Available labels: happy, angry, surprise, worried, sad, disgust. Write your detailed reasoning in <think>, and the final decision in <answer>.",
            "You are asked to classify the speaker’s emotion using a multimodal reasoning process. Choose only one from: happy, angry, surprise, worried, sad. Explain your decision using <think>, citing the person’s expression, voice, and language. End with the emotion label in <answer>.",
            "Given the visual and auditory input in the video, determine the person’s emotional state. Select one from: happy, angry, surprise, worried, sad. Justify your answer using step-by-step reasoning within <think>. Your final emotion label should go inside <answer>.",
        ]      

        self.task_pool = ["emotion", "think", "reason"]
        
        self.annotation = self.annotation[self.annotation['MER2023_mode'].isin(['train'])]

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
        text = row.get("english", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
            
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
            if answer != "neutral":
                if random.random() < 0.5:
                    instruction_pool = self.emotion_instruction_pool_no_neutral_class
                else:
                    instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            instruction_pool = self.multimodal_instruction_pool 
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.multimodal_instruction_pool
            if str(label) != "neutral":
                if random.random() < 0.8:
                    instruction_pool = self.think_instruction_pool_no_neutral_class


        if self._check_answer(answer):
            task = "emotion"
            answer = label
            instruction_pool = self.emotion_instruction_pool            
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                  
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }
        

class CAERDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Based on the visual and audio content of the video, identify the emotion expressed by the person. Choose one label from: happy, anger, neutral, surprise, fear, sad, disgust.",
            "Please determine the emotional state conveyed in the video using both visual and auditory cues. Select only one emotion from the following list: neutral, anger, disgust, sad, happy, fear, surprise.",
            "From the video's multimodal information, what is the most likely emotional category being expressed? Choose one of the following: fear, disgust, anger, surprise, happy, sad, neutral.",
            "Using both audio and visual signals from the video, classify the person's emotion into one of these categories: anger, surprise, happy, fear, neutral, disgust, sad.",
            "Analyze the video input and select the best matching emotional label from the following: sad, happy, disgust, neutral, surprise, anger, fear.",
            "Considering both what you see and hear in the video, determine the individual's emotional state. Choose one of: disgust, happy, fear, neutral, sad, surprise, anger.",
            "Your task is to classify the emotional expression shown in the video, based on both audio and visual input. Select one label from: surprise, anger, fear, happy, disgust, neutral, sad.",
            "Please evaluate the emotion expressed by the person in the video using multimodal cues, and choose one of these labels: neutral, sad, fear, happy, surprise, anger, disgust.",
            "Watch the video and identify which emotion is being portrayed. Your answer must be a single label from: anger, disgust, sad, fear, happy, neutral, surprise.",
            "Given the visual and audio signals in the video, select the most appropriate emotion from the list: surprise, sad, happy, anger, fear, neutral, disgust.",
        ]
        
        self.think_instruction_pool = [
            "Based on the video’s combined visual, auditory, and textual cues, infer the emotional state of the person. Choose one emotion from: happy, anger, neutral, surprise, fear, sad, disgust.",
            "Please determine the person’s emotional state by analyzing their facial expressions, body language, tone of voice, and spoken content. Select one of the following labels: neutral, anger, disgust, sad, happy, fear, surprise.",
            "From the video’s multimodal content—including visual appearance, vocal characteristics, and verbal meaning—identify the most appropriate emotion: fear, disgust, anger, surprise, happy, sad, neutral.",
            "Using the full spectrum of information in the video (facial behavior, gestures, intonation, and spoken words), classify the person’s emotion into one of the following categories: anger, surprise, happy, fear, neutral, disgust, sad.",
            "Analyze the video input holistically—consider how the person looks, sounds, and what they say—and choose the emotion that best matches from: sad, happy, disgust, neutral, surprise, anger, fear.",
            "Considering the visual scene, vocal delivery, and semantic content of the person’s speech, determine their emotional state. Select one from: disgust, happy, fear, neutral, sad, surprise, anger.",
            "Your task is to identify the emotion being expressed in the video by interpreting visual, auditory, and textual features. Choose only one label from: surprise, anger, fear, happy, disgust, neutral, sad.",
            "Evaluate the multimodal signals in the video—expression, posture, tone, and language—and choose the label that best reflects the person’s emotion: neutral, sad, fear, happy, surprise, anger, disgust.",
            "Watch and listen to the video, and based on the person’s expressions, voice, and what is said, select the single most likely emotional label from: anger, disgust, sad, fear, happy, neutral, surprise.",
            "Given the facial cues, vocal tone, and semantic content in the video, reason about the emotional state being expressed. Choose one of: surprise, sad, happy, anger, fear, neutral, disgust.",
        ]

        self.reason_instruction_pool = [
            "You are given a video with both visual and auditory information. The possible emotion labels are: happy, anger, neutral, surprise, fear, sad, and disgust. Based on the person's facial expressions, body language, tone of voice, and speech content, determine which emotion is being expressed and explain your reasoning.",
            "Possible emotion categories: happy, anger, neutral, surprise, fear, sad, disgust. Analyze the video by considering visual cues (such as expressions and movements), audio signals (such as tone and pitch), and what is being said. Identify the most appropriate emotion and justify your choice.",
            "The emotion expressed in the video must be one of the following: happy, anger, neutral, surprise, fear, sad, disgust. Please reason through the visual (e.g., face, posture), auditory (e.g., tone, prosody), and textual (spoken words) aspects to determine which emotion is most evident. Then, provide your final emotion label with an explanation.",
            "Emotion options: happy, anger, neutral, surprise, fear, sad, disgust. Watch the video and integrate the person’s expressions, vocal delivery, and speech to infer their emotional state. Conclude with one emotion label and explain how each modality contributed to your decision.",
            "Choose one emotion from the following list: happy, anger, neutral, surprise, fear, sad, disgust. Your task is to analyze the video input from a multimodal perspective—looking at the facial dynamics, gestures, vocal tone, and speech semantics—and determine which emotion is being conveyed. Provide a clear explanation for your choice.",
            "The emotional label must be selected from: happy, anger, neutral, surprise, fear, sad, disgust. Please analyze the visual and auditory elements in the video, including the speaker’s facial expressions, gestures, vocal tone, and language. Then decide the most likely emotion and explain why.",
            "Possible emotions: happy, anger, neutral, surprise, fear, sad, disgust. Based on the multimodal features in the video, such as visual appearance, vocal behavior, and speech content, identify the expressed emotion and describe your reasoning process in detail.",
            "Select one of the following emotion labels: happy, anger, neutral, surprise, fear, sad, disgust. Watch the video and analyze the person’s non-verbal signals, tone of voice, and spoken words. Infer the most suitable emotional category and explain what led you to that conclusion.",
            "You must classify the emotion as one of: happy, anger, neutral, surprise, fear, sad, disgust. Using the visual and audio context in the video, determine the expressed emotion and justify your reasoning with evidence from facial, vocal, and verbal signals.",
            "Emotion choices: happy, anger, neutral, surprise, fear, sad, disgust. Observe the visual and auditory features in the video and use them to reason about the person's emotional state. Conclude with a single emotion label and explain your judgment based on multimodal evidence.",
        ]


        self.task_pool = ["emotion", "reason", "think"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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
        text = row.get("text_whisper", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.think_instruction_pool
            
        if self._check_answer(answer):
            task = "emotion"
            answer = label
            instruction_pool = self.emotion_instruction_pool 
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                        
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }

        
class DFEWDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Based on what you see and hear in the video, what emotion is the person showing? Pick just one from: anger, neutral, surprise, fear, happy, disgust, or sad.",
            "Look at the video closely and, using both the visuals and the audio, tell me: is the person feeling happy, or angry, or maybe neutral, surprised, fearful, sad, or disgusted?",
            "From all the visual and auditory cues in the video, which one emotion best describes how the person feels? Choose among: fear, surprise, sad, happy, anger, disgust, or neutral.",
            "Using both what you hear and see in the video, figure out the person's emotion. Is it one of the following: surprise, disgust, sad, happy, neutral, fear, or anger?",
            "After watching the video, which of these best fits the emotion the person is expressing: sad, fear, neutral, disgust, happy, surprise, or anger?",
            "Taking into account both facial expressions and tone of voice in the video, what single emotion do you recognize—anger, disgust, happy, sad, fear, neutral, or surprise?",
            "Your task is to figure out the emotional tone of the video. Pick one label that fits best from this list: surprise, anger, happy, neutral, fear, sad, or disgust.",
            "Watch the video and choose the one emotion that seems to match the person's expression and voice. It could be fear, anger, happy, sad, neutral, surprise, or disgust.",
            "From the video's multimodal signals, what's your interpretation of the person's emotional state? Choose only one: happy, sad, disgust, surprise, anger, neutral, or fear.",
            "Based on everything shown and heard in the video, which of these emotions does the person most likely feel: neutral, happy, surprise, sad, fear, anger, or disgust?",
        ]
        
        self.emotion_instruction_pool_no_neutral_class = [
            "Based on what you see and hear in the video, what emotion is the person showing? Pick just one from: anger, surprise, fear, happy, disgust, or sad.",
            "Look at the video closely and, using both the visuals and the audio, tell me: is the person feeling happy, or angry, or maybe surprised, fearful, sad, or disgusted?",
            "From all the visual and auditory cues in the video, which one emotion best describes how the person feels? Choose among: fear, surprise, sad, happy, anger or disgust.",
            "Using both what you hear and see in the video, figure out the person's emotion. Is it one of the following: surprise, disgust, sad, happy, neutral, fear, or anger?",
            "After watching the video, which of these best fits the emotion the person is expressing: sad, fear, disgust, happy, surprise, or anger?",
            "Taking into account both facial expressions and tone of voice in the video, what single emotion do you recognize—anger, disgust, happy, sad, fear, or surprise?",
            "Your task is to figure out the emotional tone of the video. Pick one label that fits best from this list: surprise, anger, happy, fear, sad, or disgust.",
            "Watch the video and choose the one emotion that seems to match the person's expression and voice. It could be fear, anger, happy, sad, surprise, or disgust.",
            "From the video's multimodal signals, what's your interpretation of the person's emotional state? Choose only one: happy, sad, disgust, surprise, anger, or fear.",
            "Based on everything shown and heard in the video, which of these emotions does the person most likely feel: happy, surprise, sad, fear, anger, or disgust?",
        ]
        
        self.reason_instruction_pool = [
            "The possible emotions are: happy, anger, neutral, surprise, fear, sad, and disgust. Based on what you see and hear in the video—including facial expressions, gestures, vocal tone, and spoken words—identify the emotion the person is expressing and explain which cues led to your conclusion.",
            "Choose one emotion from the following list: happy, anger, neutral, surprise, fear, sad, disgust. Watch the video and use both visual (e.g., face, posture, movement) and auditory (e.g., tone, intonation, speech) signals to infer the person’s emotional state. Please describe your reasoning process clearly.",
            "Possible emotion categories: happy, anger, neutral, surprise, fear, sad, disgust. From the video’s multimodal content—what is seen and heard—determine which emotion is most likely being expressed. Justify your choice by referring to specific cues.",
            "Based on both visual and auditory information from the video, which emotion best describes how the person feels? Choose from: happy, anger, neutral, surprise, fear, sad, disgust. Explain your reasoning by identifying the key features that informed your decision.",
            "The emotional label must be one of: happy, anger, neutral, surprise, fear, sad, disgust. Taking into account the person's facial expressions, body movements, tone of voice, and speech content, identify their emotional state and explain the observable clues behind your conclusion.",
            "Select one emotion from this list: happy, anger, neutral, surprise, fear, sad, disgust. Carefully examine the person’s facial behavior and vocal delivery in the video. Then, describe your inference process and explain which cues led you to the chosen label.",
            "Emotion options: happy, anger, neutral, surprise, fear, sad, disgust. Analyze the video using a multimodal approach—looking at visual expressions, body language, voice tone, and what is being said. Determine the most likely emotion and justify your decision based on these observations.",
            "Watch the video and identify which one emotion—among happy, anger, neutral, surprise, fear, sad, disgust—is most accurately reflected by the person’s expression and vocal behavior. Explain how visual and audio cues together shaped your interpretation.",
            "You must choose from the following emotions: happy, anger, neutral, surprise, fear, sad, disgust. Use all available visual and auditory information to interpret the person's emotional state, and explain your reasoning with reference to key multimodal evidence.",
            "After watching the video, determine which one of the following emotions the person is most likely expressing: happy, anger, neutral, surprise, fear, sad, disgust. Justify your answer by analyzing facial cues, body posture, tone of voice, and speech content.",
        ]

        self.think_instruction_pool = [
            "You are given a video containing both visual and audio information. The possible emotion categories are: happy, anger, neutral, surprise, fear, sad, disgust. First, analyze the video by reasoning through facial expressions, gestures, tone of voice, and what is said. Write your reasoning inside the <think> and </think> tags. Then, select the most appropriate emotion and place it inside the <answer> and </answer> tags.",
            "Watch the video and consider all visual and auditory cues—facial expressions, body movements, voice pitch and tempo, and speech content. The emotion must be one of: happy, anger, neutral, surprise, fear, sad, disgust. Use <think> to explain your reasoning step by step, then put the final emotion label in <answer>.",
            "Analyze the multimodal signals in the video. The goal is to infer the person’s emotional state from the following options: happy, anger, neutral, surprise, fear, sad, disgust. In the <think> section, describe how visual and audio evidence supports your reasoning. Then write the most likely emotion label in <answer>.",
            "You need to determine which one of the following emotions is being expressed: happy, anger, neutral, surprise, fear, sad, disgust. Begin by reasoning through the video’s visual and auditory cues. Write this reasoning in <think>. Then, provide your chosen emotion in <answer>.",
            "Your task is to infer the emotion shown in the video using both what you see and hear. Possible categories: happy, anger, neutral, surprise, fear, sad, disgust. First, reason through the video in the <think> section. Include details such as facial expressions, body posture, tone, and wording. Then conclude with a final label in <answer>.",
            "Based on the speaker’s visual and auditory behavior in the video, choose one emotion from: happy, anger, neutral, surprise, fear, sad, disgust. In <think>, describe the visual cues (e.g., eye contact, gestures), the vocal tone (e.g., stressed, calm), and the meaning of what is said. Then, in <answer>, provide your final label.",
            "The video provides multimodal emotional cues. Consider the person’s expression, movement, tone, and speech. Possible emotion labels are: happy, anger, neutral, surprise, fear, sad, disgust. Use <think> to perform step-by-step reasoning. Then, select the most suitable label and place it in <answer>.",
            "Watch and analyze the video content carefully. Take into account facial movements, body dynamics, vocal inflections, and linguistic content. Available labels: happy, anger, neutral, surprise, fear, sad, disgust. Write your detailed reasoning in <think>, and the final decision in <answer>.",
            "You are asked to classify the speaker’s emotion using a multimodal reasoning process. Choose only one from: happy, anger, neutral, surprise, fear, sad, disgust. Explain your decision using <think>, citing the person’s expression, voice, and language. End with the emotion label in <answer>.",
            "Given the visual and auditory input in the video, determine the person’s emotional state. Select one from: happy, anger, neutral, surprise, fear, sad, disgust. Justify your answer using step-by-step reasoning within <think>. Your final emotion label should go inside <answer>.",
        ]
        self.think_instruction_pool_no_neutral_class = [
            "You are given a video containing both visual and audio information. The possible emotion categories are: happy, anger, surprise, fear, sad, disgust. First, analyze the video by reasoning through facial expressions, gestures, tone of voice, and what is said. Write your reasoning inside the <think> and </think> tags. Then, select the most appropriate emotion and place it inside the <answer> and </answer> tags.",
            "Watch the video and consider all visual and auditory cues—facial expressions, body movements, voice pitch and tempo, and speech content. The emotion must be one of: happy, anger, doubt, surprise, fear, sad, disgust. Use <think> to explain your reasoning step by step, then put the final emotion label in <answer>.",
            "Analyze the multimodal signals in the video. The goal is to infer the person’s emotional state from the following options: happy, anger, contempt, surprise, fear, sad, disgust. In the <think> section, describe how visual and audio evidence supports your reasoning. Then write the most likely emotion label in <answer>.",
            "You need to determine which one of the following emotions is being expressed: happy, anger, worry, surprise, fear, sad, disgust. Begin by reasoning through the video’s visual and auditory cues. Write this reasoning in <think>. Then, provide your chosen emotion in <answer>.",
            "Your task is to infer the emotion shown in the video using both what you see and hear. Possible categories: happy, anger, surprise, fear, sad, disgust. First, reason through the video in the <think> section. Include details such as facial expressions, body posture, tone, and wording. Then conclude with a final label in <answer>.",
            "Based on the speaker’s visual and auditory behavior in the video, choose one emotion from: happy, anger, surprise, fear, sad, disgust. In <think>, describe the visual cues (e.g., eye contact, gestures), the vocal tone (e.g., stressed, calm), and the meaning of what is said. Then, in <answer>, provide your final label.",
            "The video provides multimodal emotional cues. Consider the person’s expression, movement, tone, and speech. Possible emotion labels are: happy, anger, doubt, surprise, fear, sad, disgust. Use <think> to perform step-by-step reasoning. Then, select the most suitable label and place it in <answer>.",
            "Watch and analyze the video content carefully. Take into account facial movements, body dynamics, vocal inflections, and linguistic content. Available labels: happy, anger, contempt, surprise, fear, sad, disgust. Write your detailed reasoning in <think>, and the final decision in <answer>.",
            "You are asked to classify the speaker’s emotion using a multimodal reasoning process. Choose only one from: happy, anger, worry, surprise, fear, sad, disgust. Explain your decision using <think>, citing the person’s expression, voice, and language. End with the emotion label in <answer>.",
            "Given the visual and auditory input in the video, determine the person’s emotional state. Select one from: happy, anger, surprise, fear, sad, disgust. Justify your answer using step-by-step reasoning within <think>. Your final emotion label should go inside <answer>.",
        ]
        
        self.task_pool = ["emotion", "reason", "think"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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

        label = row.get("label", "unknown")
        text = row.get("sentence", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
            if answer != "neutral":
                if random.random() < 0.5:
                    instruction_pool = self.emotion_instruction_pool_no_neutral_class
                else:
                    instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.think_instruction_pool 
            if str(label) != "neutral":
                if random.random() < 0.5:
                    instruction_pool = self.think_instruction_pool_no_neutral_class
                                
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }


class MCEIUDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Based on both what you see and hear in the video, which one emotion is the person showing? Choose from: anger, neutral, surprise, fear, happy, disgust, or sad.",
            "Watch the video carefully and consider both visuals and audio—does the person seem happy, angry, neutral, surprised, fearful, sad, or disgusted?",
            "From the video's audio and visual cues, which emotion best captures how the person feels? Select one: fear, surprise, sad, happy, anger, disgust, or neutral.",
            "Using both the sound and the visuals in the video, what emotion do you think the person is expressing? Pick one from: surprise, disgust, sad, happy, neutral, fear, or anger.",
            "After watching the video, which of the following emotions best describes what the person is feeling: sad, fear, neutral, disgust, happy, surprise, or anger?",
            "Taking into account facial expression and voice tone, which one emotion do you think the person is showing: anger, disgust, happy, sad, fear, neutral, or surprise?",
            "Your task is to identify the emotion in the video. Select the one label that best fits from: surprise, anger, happy, neutral, fear, sad, or disgust.",
            "Watch and listen carefully—what single emotion do you think the person is expressing? Choose from: fear, anger, happy, sad, neutral, surprise, or disgust.",
            "Based on the video's multimodal signals, how would you describe the person's emotional state? Select one: happy, sad, disgust, surprise, anger, neutral, or fear.",
            "Considering all the visual and audio information in the video, which of these emotions seems most likely: neutral, happy, surprise, sad, fear, anger, or disgust?",
        ]
        self.intent_instruction_pool = [
            "Based on the visual, audio, and textual cues in the video, what is the speaker's intended action or communicative goal? Choose one of: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, or wishing.",
            "Analyze the multimodal information from the video and identify the speaker's intent. Select a single label from: consoling, suggesting, agreeing, wishing, acknowledging, questioning, encouraging, or neutral.",
            "What is the most likely communicative intent being expressed in the video? Your answer must be one of the following: encouraging, agreeing, neutral, acknowledging, suggesting, wishing, questioning, or consoling.",
            "From what you can see and hear in the video—including facial expression, tone of voice, and any text—what is the person's intent? Pick one: questioning, neutral, consoling, encouraging, agreeing, suggesting, wishing, or acknowledging.",
            "Watch the video and determine the speaker's communicative purpose. Choose only one from: acknowledging, encouraging, neutral, consoling, agreeing, questioning, suggesting, or wishing.",
            "What intention is being conveyed in this video? Consider all available modalities and select one from: wishing, questioning, consoling, suggesting, encouraging, acknowledging, agreeing, or neutral.",
            "Using all available information (visuals, audio, and text), what best describes the person's communicative intent? Select one of: agreeing, neutral, acknowledging, suggesting, questioning, consoling, wishing, or encouraging.",
            "What is the speaker trying to do or communicate in the video? Pick the most appropriate label from: encouraging, questioning, suggesting, neutral, agreeing, acknowledging, consoling, or wishing.",
            "Please identify the speaker's intention based on the multimodal signals in the video. Choose from: questioning, agreeing, acknowledging, wishing, encouraging, suggesting, neutral, or consoling.",
            "From the combined visual, vocal, and textual cues in the video, what is the most likely intention behind the speaker's behavior? Your choices are: consoling, neutral, acknowledging, suggesting, wishing, agreeing, questioning, or encouraging.",
        ]

        self.think_instruction_pool = [
            "You are given a video with visual, audio, and textual information. The possible intent labels are: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. First, analyze the speaker’s behavior using all modalities and describe your reasoning inside <think> and </think>. Then, select the most appropriate intent and place it inside <answer> and </answer>.",
            "Watch the video and infer the speaker’s communicative intent. Available options: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Use <think> to explain how facial expressions, tone, and speech contributed to your interpretation. Then give your final answer in <answer>.",
            "Analyze the speaker’s intention using multimodal cues from the video—such as body language, voice tone, and spoken words. Choose one label from: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Provide step-by-step reasoning in <think> and the intent in <answer>.",
            "What is the speaker trying to express through their multimodal behavior? Possible intent categories: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Explain your thought process in <think> and finalize your choice in <answer>.",
            "Given the visual, vocal, and textual information in the video, determine the speaker’s intent. Select one from: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. First reason through the clues in <think>, then provide your label in <answer>.",
            "Your task is to infer the speaker’s communicative goal based on all available modalities. Choose one of the following intents: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Explain your reasoning in <think> and your conclusion in <answer>.",
            "What is the most likely intention behind the speaker’s words and nonverbal behavior in the video? Possible labels: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Justify your decision with multimodal reasoning in <think> and state the answer in <answer>.",
            "Use the visual, audio, and textual context of the video to reason about the speaker’s intent. Choose one label from: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Provide your reasoning in <think> and the final intent in <answer>.",
            "Watch the speaker’s expressions, body language, tone of voice, and speech content. What are they trying to communicate? Pick from: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Put your explanation in <think> and the result in <answer>.",
            "From the multimodal signals in the video, what is the speaker’s intention? Options: acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing. Analyze the clues step-by-step in <think> and conclude with <answer>.",
            "You are given a video with both visual and auditory content, including facial expressions, body language, tone of voice, and spoken words. Carefully observe the speaker’s behavior and reasoning process. Describe your interpretation of the speaker’s communicative intent in <think> and then summarize your final conclusion in <answer>.",
            "Based on the combined visual, audio, and textual information from the video, infer what the speaker is trying to convey or achieve through their behavior. Explain your reasoning step by step inside <think>, and clearly state the communicative purpose you believe they are expressing inside <answer>.",
        ]

        self.reason_instruction_pool = [
            "Given the person’s expressions, tone of voice, body language, and what they say in the video, what do you think they are trying to achieve or communicate? Please explain your reasoning.",
            "Watch the video and analyze the person’s non-verbal cues and speech. What is their underlying communicative goal or intention? Justify your interpretation with evidence from both visual and audio signals.",
            "Based on the speaker's facial expressions, gestures, vocal tone, and words, what do you infer is their intent in this moment? What are they trying to convey or accomplish?",
            "Using all available modalities—visuals, audio, and language—describe what the character is trying to express or do. Focus on their intent rather than just their emotion.",
            "Observe the person’s behavior in the video: how they speak, move, and express themselves. What do these cues reveal about their purpose or communicative intention?",
            "What do you think the speaker’s goal is in this interaction? Consider facial expressions, body posture, speech tone, and context clues from the video.",
            "From the way the speaker interacts visually and vocally, what kind of response or effect do you think they are aiming for? What is their role in the conversation?",
            "By examining how the person looks, speaks, and acts in the video, infer their intended purpose. Are they trying to inform, comfort, challenge, agree, or something else?",
        ]

        
        self.task_pool = ["emotion", "reason", "think", "intent", "intent"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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
        intent_label = row.get("intent", "unknown")
        text = row.get("Subtitle", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
        elif task == "intent":
            answer = intent_label
            instruction_pool = self.intent_instruction_pool     
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_intent", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_intent", ""))
            answer = "<think>" + answer + "</think>" + "<answer>" + intent_label + "</answer>"
            instruction_pool = self.think_instruction_pool


        if self._check_answer(answer):
            task = "intent"
            answer = intent_label
            instruction_pool = self.intent_instruction_pool     
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                               
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }


class E3DatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Based on the combined visual, auditory, and textual cues in the video, what emotion is the person experiencing? Choose one of: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised.",
            "Watch the video and analyze the person's facial expressions, tone of voice, and spoken words. What emotion are they expressing? Select one: scared, sarcastic, happy, surprised, shy, anger, sad, or disgusted.",
            "What is the most likely emotional state shown in the video? Consider all modalities and pick one from: surprised, happy, disgusted, scared, sarcastic, shy, sad, or anger.",
            "Please identify the person's emotion based on multimodal information. Your answer must be one of the following: shy, disgusted, sarcastic, anger, surprised, sad, scared, or happy.",
            "From what you observe and hear in the video, how is the person feeling? Choose only one emotion from: anger, happy, scared, shy, disgusted, sarcastic, surprised, or sad.",
            "Given the visual, audio, and text input, what best describes the emotional state of the individual? Pick one from: sarcastic, anger, shy, disgusted, surprised, sad, scared, or happy.",
            "Your task is to recognize the emotional state being displayed in the video. Choose a single label from: happy, disgusted, shy, scared, sad, anger, sarcastic, or surprised.",
            "What emotion is most clearly expressed by the speaker in this video? Consider all available cues and select one of: surprised, sarcastic, scared, shy, happy, anger, sad, or disgusted.",
            "From the person's tone, facial expression, and the context of their words, what emotion are they most likely feeling? Choose one: sad, sarcastic, happy, shy, surprised, anger, disgusted, or scared.",
            "Analyze the full video content—visuals, voice, and language—and determine the emotional state being conveyed. Your options are: disgusted, sarcastic, anger, scared, surprised, sad, shy, or happy.",
        ]
        self.think_instruction_pool = [
            "The available emotion categories are: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised. Watch the video and analyze the person’s facial expression, vocal cues, and spoken content. Write your reasoning in <think>, and give your final answer in <answer>.",
            "After observing the person's behavior in the video, including tone, expressions, and speech, what do you think their emotional state is? Choose from the following: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised. Justify your answer in <think> and select a label in <answer>.",
            "Analyze the multimodal signals—such as facial gestures, prosody, and spoken words—in the video. Your possible choices are: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised. Provide your thought process in <think> and your conclusion in <answer>.",
            "Watch the video and determine the person’s emotional state. Based on visual, audio, and textual evidence, explain your reasoning step-by-step in <think>. Then, pick one from the following labels to complete <answer>: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised.",
            "Look closely at the person’s eye movements, posture, tone of voice, and verbal content. What emotion are they conveying? First reason it out in <think>, then write your choice—angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised—in <answer>.",
            "Emotion categories include angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised. Your task is to identify which one fits best given the speaker’s multimodal behavior. Provide your explanation in <think> and the emotion label in <answer>.",
            "The person’s facial expressions, speech rhythm, and word choice all offer clues about their emotional state. What emotion are they experiencing? (Options: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised.) Explain in <think>, and write the label in <answer>.",
            "From the visual and auditory context in the video, determine the dominant emotion the person expresses. Be sure to think aloud in <think>. The final answer must be one of: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised—written in <answer>.",
            "Based on what you see and hear in the video, including body language and vocal delivery, identify the speaker's emotional state. In <think>, explain how you reached your decision. Then choose one of these: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised and enter it in <answer>.",
            "Facial expressions, tone, and speech can all reveal how someone feels. Think through what these cues suggest in <think>. The options are: angry, disgusted, happy, sad, sarcastic, scared, shy, or surprised. Write your final label in <answer>.",
        ]
        
        self.task_pool = ["emotion", "think", "reason"]

        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

    def __getitem__(self, index):
        try:
            row = self.annotation.iloc[index]
            video_id = row["video_id"]
            
            image = None
            video_path = os.path.join(self.vis_root, f"{video_id}.mp4")
            if os.path.exists(video_path):
                image = self._extract_frame(video_path)     
            image = Image.fromarray(image.astype('uint8'))
            image = image.convert('RGB')
            image = self.vis_processor(image)   

            label = row.get("emotion", "unknown")
            text = row.get("conversation", "")
            if not isinstance(text, str):
                text = str(text) if not (text != text) else "" 
            person = row.get("person", "")      

            # Whisper feature
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            
            # EVA-ViT-G
            video_features = self._load_npy(video_id, "EVA-ViT-G")

            # random task
            task = random.choice(self.task_pool)
            if task == "emotion":
                answer = label
                instruction_pool = self.emotion_instruction_pool
            elif task == "reason":
                answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
                instruction_pool = self.multimodal_instruction_pool
            elif task == "think":
                answer = str(self._safe_str(row.get("gpt-4o-mini_Emotion", "")))
                answer = "<think>" + answer + "</think>" + "<answer>" + str(label) + "</answer>"
                instruction_pool = self.think_instruction_pool 

            if self._check_answer(answer):
                task = "emotion"
                answer = label
                instruction_pool = self.emotion_instruction_pool  
                audio_features = self._load_npy(video_id, "whisper-large-v3")
                video_features = self._load_npy(video_id, "EVA-ViT-G")            
                                
            character_line = "{} in video says: {}. ".format(person, text)
            instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

            if not isinstance(answer, str):
                answer = str(answer) if not (answer != answer) else "" 
            answer = self.text_processor(answer)
            return {
                "image": image,
                "video_features": video_features,
                "audio_features": audio_features,
                "video_id": video_id,
                "text": text,
                "instruction_input": instruction,
                "answer": answer,
            }
        except Exception as e:
            print(f"\n❌ ERROR in sample {index} | video_id = {video_id}")
            print("Exception type:", type(e).__name__)
            print("Exception message:", e)
        
        
class IEMOCAPDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_3_class_instruction_pool = [
            "Based on the video, what is the emotional state of the person? Choose one from: Happiness, Sadness, or Anger.",
            "Please watch the video and identify the person's emotion. Is it one of these: Sadness, Anger, or Happiness?",
            "Considering all available cues—visuals, audio, and text—select the most fitting emotion from the following list: Anger, Sadness, Happiness.",
            "What emotion is the individual in the video most likely experiencing? Pick just one: Happiness, Sadness, or Anger.",
            "Look at the person's expressions, tone of voice, and what they say. Which of these emotions best describes their current state: Anger, Happiness, or Sadness?",
            "Choose the best label to describe the emotional state shown in the video. Options: Sadness, Happiness, Anger.",
            "From the multimodal input, determine the emotion being expressed. Your answer must be one of the following: Happiness, Sadness, Anger.",
            "Please determine the emotional tone of the speaker using all modalities. Select from: Anger, Happiness, or Sadness.",
            "Analyze the video and identify which of the following emotions the person is expressing: Sadness, Anger, Happiness.",
            "Given the person's behavior and expression in the video, what is their emotional state? Options include: Happiness, Anger, and Sadness.",
        ]
        self.emotion_4_class_instruction_pool = [ # Neutral, Happiness, Sadness, Anger
            "Based on the video, what is the emotional state of the person? Choose one from: Neutral, Happiness, Sadness, or Anger.",
            "Please watch the video and identify the person's emotion. Is it one of these: Sadness, Anger, Neutral, or Happiness?",
            "Considering all available cues—visuals, audio, and text—select the most fitting emotion from the following list: Anger, Sadness, Happiness, Neutral.",
            "What emotion is the individual in the video most likely experiencing? Pick just one: Happiness, Sadness, Neutral, or Anger.",
            "Look at the person's expressions, tone of voice, and what they say. Which of these emotions best describes their current state: Neutral, Anger, Happiness, or Sadness?",
            "Choose the best label to describe the emotional state shown in the video. Options: Sadness, Happiness, Anger, Neutral.",
            "From the multimodal input, determine the emotion being expressed. Your answer must be one of the following: Happiness, Neutral, Sadness, Anger.",
            "Please determine the emotional tone of the speaker using all modalities. Select from: Anger, Neutral, Happiness, or Sadness.",
            "Analyze the video and identify which of the following emotions the person is expressing: Neutral, Sadness, Anger, Happiness.",
            "Given the person's behavior and expression in the video, what is their emotional state? Options include: Happiness, Anger, Neutral, and Sadness.",
        ]
        self.emotion_instruction_pool = [
            "Based on the video's visual, audio, and textual cues, which of the following emotions is the person experiencing: Anger, Happiness, Disgust, Surprise, Fear, Sadness, Excited, Neutral, or Frustration?",
            "What is the emotional state being expressed in the video? Please choose one from the following options: Neutral, Frustration, Anger, Excited, Happiness, Surprise, Sadness, Fear, or Disgust.",
            "Watch and listen to the video carefully. Based on what you observe, what is the person's emotion? Pick one: Sadness, Excited, Neutral, Fear, Disgust, Surprise, Happiness, Anger, or Frustration.",
            "Analyze the person's emotional expression using all available modalities. Select a label from: Happiness, Anger, Fear, Sadness, Excited, Neutral, Frustration, Surprise, or Disgust.",
            "Considering facial expression, tone of voice, and speech, which of the following emotions does the individual show: Fear, Surprise, Disgust, Excited, Happiness, Frustration, Anger, Neutral, or Sadness?",
            "Your task is to determine the person's emotional state using all available visual and audio clues. Choose one: Frustration, Sadness, Happiness, Surprise, Disgust, Fear, Neutral, Anger, or Excited.",
            "What is the most likely emotional category represented in the video? Please choose one of the following: Excited, Disgust, Surprise, Happiness, Anger, Fear, Neutral, Sadness, or Frustration.",
            "From the combined signals in the video (visual, voice, and text), which emotion best describes the person? Options are: Sadness, Anger, Neutral, Happiness, Frustration, Surprise, Excited, Fear, or Disgust.",
            "Review the multimodal cues in the video and decide which of the following best describes the speaker's emotional state: Disgust, Fear, Excited, Surprise, Sadness, Neutral, Anger, Happiness, or Frustration.",
            "Given the expressions, tone, and speech in the video, identify the dominant emotion. Select from: Neutral, Disgust, Surprise, Happiness, Anger, Frustration, Sadness, Excited, or Fear.",
        ]
        
        self.think_instruction_pool = [
            "The possible emotions are: Anger, Happiness, Disgust, Surprise, Fear, Sadness, Excited, Neutral, Frustration. Analyze the person's emotional state in the video by combining visual, auditory, and textual clues. Provide your reasoning in <think> and your final label in <answer>.",
            "Watch the video closely and determine the emotional category being expressed. Use <think> to explain your reasoning based on tone, gestures, and words. Then in <answer>, choose one from: Surprise, Fear, Frustration, Happiness, Disgust, Anger, Sadness, Excited, Neutral.",
            "Based on the multimodal information in the video, what emotion is being shown? Categories to choose from: Neutral, Fear, Sadness, Frustration, Happiness, Surprise, Anger, Excited, Disgust. Think through your interpretation in <think> and decide in <answer>.",
            "What is the dominant emotional state expressed by the speaker in this video? Consider expressions, tone, and speech. Use <think> to explain your reasoning. Your final label must be one of: Happiness, Sadness, Frustration, Neutral, Disgust, Excited, Surprise, Fear, Anger, placed in <answer>.",
            "Determine the person’s emotion from the video. First reflect on their multimodal behavior in <think>, then pick a label from: Disgust, Excited, Frustration, Happiness, Neutral, Sadness, Anger, Fear, Surprise and write it in <answer>.",
            "From the way the person speaks, looks, and gestures, what emotion are they likely feeling? Please reason in <think>, then choose your final label in <answer>. Possible options: Surprise, Disgust, Frustration, Anger, Sadness, Happiness, Fear, Neutral, Excited.",
            "Emotion choices: Sadness, Excited, Happiness, Neutral, Fear, Disgust, Surprise, Frustration, Anger. Analyze the speaker’s tone, expression, and words to infer their emotional state. Explain in <think> and conclude in <answer>.",
            "Using cues from all modalities—visuals, audio, and spoken content—identify the person’s emotion. Write your analysis in <think> and provide your answer in <answer>. Options include: Fear, Anger, Surprise, Excited, Disgust, Neutral, Happiness, Frustration, Sadness.",
            "Review the person's expressions and tone in the video. What emotion are they trying to express? Use <think> to describe your reasoning, and in <answer> choose one from the list: Anger, Disgust, Excited, Happiness, Neutral, Surprise, Frustration, Fear, Sadness.",
            "The speaker's behavior in the video may reflect one of the following: Frustration, Anger, Surprise, Sadness, Neutral, Excited, Happiness, Disgust, or Fear. Think carefully in <think> and select your final answer in <answer>.",
        ]
        
        self.task_pool = ["emotion", "think", "reason"]
        
        self.annotation = self.annotation[self.annotation['session'].isin(['Session1', 'Session2', 'Session3', 'Session4'])]

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

        label = row.get("official_label", "unknown")
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            if answer in ['Anger', 'Sadness', 'Happiness']:
                if random.random() < 0.5:
                    instruction_pool = self.emotion_3_class_instruction_pool
                else:
                    instruction_pool = self.emotion_4_class_instruction_pool
            elif answer in ['Neutral']:
                if random.random() < 0.5:
                    instruction_pool = self.emotion_4_class_instruction_pool
                else:
                    instruction_pool = self.emotion_instruction_pool
            else:
                instruction_pool = self.emotion_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            instruction_pool = self.multimodal_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.think_instruction_pool 

        if self._check_answer(answer):
            task = "emotion"
            answer = label
            if answer in ['Anger', 'Sadness', 'Happiness']:
                instruction_pool = self.emotion_3_class_instruction_pool
            else:
                instruction_pool = self.emotion_instruction_pool    
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                                           
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }        
        
      
class MAFWDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Based on the video's audio, visuals, and any spoken content, what emotion is being conveyed? Please choose one from: sadness, contempt, happiness, anger, surprise, helplessness, disgust, disappointment, fear, neutral, or anxiety.",
            "From what you see and hear in the video, which of the following emotions best matches the person's behavior? Pick one: fear, anger, happiness, neutral, helplessness, anxiety, surprise, contempt, sadness, disgust, or disappointment.",
            "Please analyze the multimodal input from the video and identify the person's emotional state. Select one of the following: anxiety, sadness, surprise, contempt, happiness, neutral, fear, disappointment, anger, disgust, or helplessness.",
            "Watch the video and determine which single emotion best describes the speaker's internal state. Choose from: disappointment, disgust, fear, helplessness, sadness, happiness, neutral, contempt, surprise, anger, or anxiety.",
            "Considering visual cues, voice tone, and language in the video, which of the following emotional states is being expressed: happiness, contempt, surprise, anxiety, anger, disappointment, sadness, neutral, disgust, fear, or helplessness?",
            "After watching the video, what is your interpretation of the individual's emotion? Pick only one: contempt, fear, sadness, surprise, helplessness, neutral, happiness, anxiety, anger, disappointment, or disgust.",
            "Based on the full video context—what the person looks like, sounds like, and says—what emotion are they showing? Select one from: anxiety, anger, disappointment, fear, surprise, sadness, contempt, happiness, neutral, helplessness, or disgust.",
            "What emotional state is being communicated in this video clip? Consider all modalities and choose from: disgust, sadness, helplessness, fear, anger, anxiety, disappointment, surprise, neutral, contempt, or happiness.",
            "Review the speaker's tone, facial expression, and verbal content. What do you think is their emotional state? Options: neutral, helplessness, disappointment, happiness, anxiety, sadness, surprise, anger, contempt, fear, or disgust.",
            "Given the multimodal information in the video, what is the most likely emotion the person is experiencing? Choose one from: sadness, anger, fear, disappointment, anxiety, surprise, happiness, helplessness, disgust, contempt, or neutral.",
        ]
        self.emotion_multi_label_instruction_pool = [
            "Based on the multimodal content of the video (visuals, audio, and language), identify all emotions the person may be expressing. You may choose multiple labels. Your output can be any combination of valid emotion labels, such as: 'disgust,contempt', 'sadness,anxiety', 'anger,surprise', 'happiness,contempt', or 'anxiety,helplessness'. Use only labels from this list: anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise.",
            "Please analyze the video and predict all emotional states conveyed by the individual. This is a multi-label task, so you can select more than one emotion. For example: 'surprise,neutral', 'sadness,helplessness', or 'anger,disgust'. Only use labels from: anxiety, surprise, sadness, anger, happiness, neutral, contempt, disappointment, disgust, fear, helplessness.",
            "From the combined visual, audio, and textual information in the video, list all emotions you observe. Multiple emotions may co-occur. Your response should be a comma-separated list, like 'fear,anxiety' or 'sadness,disappointment'. Valid labels: happiness, sadness, fear, neutral, helplessness, surprise, contempt, anxiety, anger, disappointment, disgust.",
            "What emotions does the person appear to be experiencing in the video? You may select several that apply. Answer format: 'emotion1,emotion2,...'. For instance: 'anger,anxiety', 'disgust,disappointment', or 'sadness,neutral'. Use only these: disgust, fear, sadness, helplessness, happiness, anger, neutral, surprise, anxiety, disappointment, contempt.",
            "Please review the video and list any emotions being expressed. This is a multi-label classification task—more than one emotion may apply. Your answer could be: 'surprise,anxiety', 'happiness,contempt', or 'sadness,helplessness'. Use only the following labels: anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise.",
            "Using all available video input (face, voice, and text), identify the emotional states present. You can respond with multiple labels, separated by commas—for example, 'disgust,anger', or 'fear,helplessness'. Only use labels from this list: surprise, anger, disgust, happiness, sadness, neutral, contempt, anxiety, disappointment, fear, helplessness.",
            "What combination of emotions best describes the person's state in the video? You may choose several labels. Valid examples include: 'anger,surprise', 'happiness,disgust', or 'neutral,anxiety'. Only pick from: sadness, happiness, fear, disgust, anger, contempt, anxiety, helplessness, neutral, disappointment, surprise.",
            "Analyze the person's emotional state in this video. Multiple emotions may be involved. Your answer should be a combination like 'anxiety,helplessness', or 'sadness,disappointment'. Choose only from: anger, contempt, anxiety, disappointment, disgust, fear, sadness, surprise, happiness, helplessness, neutral.",
            "What emotions are present in the video? You can include more than one if needed. Format your answer like: 'fear,surprise' or 'sadness,anger'. Use only labels from: anxiety, surprise, sadness, helplessness, happiness, disgust, disappointment, fear, neutral, contempt, anger.",
            "Given the multimodal input in the video, identify the emotional states being conveyed. This is a multi-label task—feel free to list more than one. Example outputs: 'happiness,contempt', 'anxiety,disgust', or 'sadness,helplessness'. Only choose from: sadness, anger, fear, disappointment, anxiety, surprise, happiness, helplessness, disgust, contempt, neutral.",
        ]
        
        self.think_instruction_pool = [
            "The emotion categories are: sadness, contempt, happiness, anger, surprise, helplessness, disgust, disappointment, fear, neutral, and anxiety. Based on the person's visual behavior, voice tone, and spoken content in the video, analyze their emotional state in <think> and choose the best-matching label in <answer>.",
            "From the speaker’s facial expressions, vocal delivery, and language, determine which of the following emotions best matches their state: surprise, fear, happiness, helplessness, sadness, anger, contempt, disappointment, anxiety, disgust, or neutral. Explain your reasoning in <think> and provide the final label in <answer>.",
            "Analyze the video’s multimodal input—including expressions, voice, and speech. What emotion is most evident? Use <think> for your reasoning, and select one from: sadness, anger, contempt, helplessness, fear, anxiety, happiness, disappointment, surprise, disgust, or neutral inside <answer>.",
            "Consider what you see and hear in the video. Based on that, what is the person likely feeling? Reason step-by-step in <think>, then identify the emotion from this list in <answer>: anger, surprise, neutral, happiness, disappointment, sadness, fear, anxiety, helplessness, contempt, or disgust.",
            "Based on the context of the video, reflect on the speaker’s multimodal signals and choose one emotion. Please describe your inference process in <think>, then provide the final label in <answer>. The emotion should be one of: disgust, sadness, contempt, helplessness, fear, neutral, surprise, anxiety, happiness, anger, or disappointment.",
            "Which emotional state is being expressed by the individual in the video? Think about their vocal and visual cues in <think>, then pick a label in <answer>. Choose from: contempt, surprise, fear, sadness, helplessness, disappointment, disgust, anger, anxiety, neutral, or happiness.",
            "Using all available video clues—voice, expressions, posture—analyze the emotional state of the person. Categories: fear, disgust, anger, surprise, sadness, disappointment, helplessness, happiness, anxiety, neutral, contempt. Reason in <think> and conclude with <answer>.",
            "Watch the speaker's behavior and explain in <think> what emotional clues are present. Then choose a matching label in <answer> from the following options: happiness, sadness, contempt, disappointment, fear, anger, anxiety, surprise, neutral, disgust, or helplessness.",
            "Reflect on the multimodal information presented in the clip. How does the speaker feel? Explain in <think>. Choose one from this list and write it in <answer>: anxiety, helplessness, surprise, sadness, anger, contempt, disappointment, fear, neutral, happiness, or disgust.",
            "Given the person’s expressions, voice, and what they say, reason about their emotion in <think>. Then select one final label from: disappointment, anger, sadness, disgust, anxiety, happiness, surprise, helplessness, fear, neutral, or contempt, and write it in <answer>.",
        ]
        
        self.think_multi_instruction_pool = [
            "Labels to choose from: anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise. Based on facial cues, tone of voice, and language in the video, reason through your analysis in <think> and provide all applicable emotion labels in <answer> as a comma-separated list.",
            "After reviewing the video’s audio, visual, and spoken signals, describe your emotional inference in <think>. You may list multiple emotions in <answer> using only these valid labels: fear, sadness, surprise, disgust, contempt, anger, disappointment, helplessness, anxiety, happiness, neutral.",
            "This is a multi-label task. First, analyze the person’s multimodal behavior in <think>—include body language, tone, and verbal context. Then, in <answer>, provide all fitting emotions separated by commas. Valid labels include: sadness, happiness, neutral, helplessness, fear, surprise, anger, anxiety, disgust, disappointment, contempt.",
            "From the speaker’s expression, voice, and wording, what emotional states do you observe? In <think>, explain your interpretation, and in <answer> give one or more labels from: surprise, disappointment, fear, happiness, contempt, anxiety, sadness, anger, disgust, neutral, helplessness.",
            "Use all available clues—voice pitch, expression, gesture, and text—to determine the emotional states present in this video. Write your reasoning in <think>, and your final multi-label answer in <answer> using only: anger, fear, anxiety, happiness, sadness, neutral, contempt, surprise, helplessness, disgust, disappointment.",
            "What mix of emotions do you believe the person is displaying in this clip? Use <think> to explain how you came to this conclusion. Then list one or more labels from: sadness, anger, helplessness, surprise, happiness, disappointment, fear, disgust, contempt, anxiety, neutral inside <answer>.",
            "Given the multimodal context, determine all emotional states that apply. Your reasoning belongs in <think> and your labels in <answer>. Choose from: contempt, fear, surprise, helplessness, happiness, sadness, disgust, neutral, anger, disappointment, anxiety.",
            "First, analyze the character’s behavior and multimodal signals in <think>. Then in <answer>, provide the full list of emotions being conveyed. Valid emotion labels: happiness, sadness, surprise, anxiety, anger, disgust, disappointment, contempt, fear, helplessness, neutral.",
            "Review the emotional signals across all modalities and explain your reasoning in <think>. Then provide your prediction in <answer> by selecting one or more emotions from: helplessness, anger, surprise, happiness, sadness, fear, anxiety, neutral, contempt, disgust, disappointment.",
            "You are asked to perform multi-label emotion recognition. Use <think> to reason about the emotional content of the video and list all matching emotions in <answer>. Available labels: anxiety, fear, disgust, contempt, anger, surprise, happiness, helplessness, disappointment, sadness, neutral.",
        ]

    
        self.task_pool = ["emotion", "reason", "think", "multi-label", "think-multi",]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

    def __getitem__(self, index):
        row = self.annotation.iloc[index]
        video_id = str(row["video_id"]).zfill(5)
        
        image = None
        video_path = os.path.join(self.vis_root, f"{video_id}.mp4")

        if os.path.exists(video_path):
            image = self._extract_frame(video_path)     
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)   

        single_label = row.get("single_label", "unknown")
        if pd.isna(single_label) or single_label == "":
            single_label = "unknown"
        multi_label = row.get("multi_label", "unknown")
        if pd.isna(multi_label) or multi_label == "":
            multi_label = "unknown"
        text = row.get("text_whisper", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = single_label
            instruction_pool = self.emotion_instruction_pool
        elif task == "multi-label":
            if multi_label == "unknown" or multi_label == "":
                answer = single_label
                instruction_pool = self.emotion_instruction_pool
            else:
                instruction_pool = self.emotion_multi_label_instruction_pool
                answer = multi_label
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Single", ""))
            instruction_pool = self.multimodal_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Single", ""))
            answer = "<think>" + answer + "</think>" + "<answer>" + single_label + "</answer>"
            instruction_pool = self.think_instruction_pool 
        elif task == "think-multi":
            if multi_label == "unknown" or multi_label == "":
                answer = single_label
                instruction_pool = self.emotion_instruction_pool
            else:
                answer = self._safe_str(row.get("gpt-4o-mini_Multi", ""))
                answer = "<think>" + answer + "</think>" + "<answer>" + multi_label + "</answer>"
                instruction_pool = self.think_multi_instruction_pool 
                task = "think"
            
        if self._check_answer(answer):
            task = "emotion"
            answer = single_label
            instruction_pool = self.emotion_instruction_pool 
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                                       
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        # print(video_id, instruction, answer)
        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }        
        
        
class BOLDDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_multi_label_instruction_pool = [
            "Analyze the video using all available modalities (visuals, audio, and spoken words) and identify all emotional states the person may be experiencing. Your output may include multiple emotions like: 'Sadness,Pain', 'Excitement,Confidence', or 'Fear,Disapproval'. Please use only the following labels: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "What emotional states are being conveyed in the video? You can choose more than one. Answer format: 'emotion1,emotion2,...'. For instance: 'Happiness,Excitement', 'Embarrassment,Annoyance', or 'Disconnection,Sympathy'. Use only the valid labels listed here: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "From what you can observe and hear in the video, what emotions does the person seem to be expressing? You may include multiple emotions, such as 'Anticipation,Happiness', 'Disconnection,Fear', or 'Esteem,Affection'. The valid set of labels includes: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "Please determine all emotions shown in the video using multimodal analysis. This includes facial expressions, vocal tone, and content of speech. You may respond with examples like 'Pain,Suffering', 'Surprise,Engagement', or 'Embarrassment,Fatigue'. Only use labels from the following: [Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning].",
            "What emotional states are being communicated through the person's expressions, words, and voice in the video? You may choose one or more, such as: 'Embarrassment,Sensitivity', 'Fear,Disapproval', or 'Sadness,Sympathy'. Use the full emotion label list without modifications.",
            "Watch the video and identify the emotional states conveyed by the individual. Feel free to list multiple emotions if more than one applies.",
            "From the combined visual, audio, and spoken content in the video, what emotions do you perceive? This is a multi-label task, so your answer can include several emotions.",
            "Please analyze the video and describe the person's emotional state. You may select more than one emotion if needed.",
            "Consider the full context of the video—how the person looks, speaks, and acts. What emotions are they expressing? Provide one or more relevant emotional labels.",
            "What emotions are present in the video? Your answer can include multiple categories if the person appears to be feeling more than one emotion.",
        ]

        self.reason_instruction_pool = [
            "Carefully analyze the video using all modalities (visual cues, vocal tone, and spoken language). Based on these clues, infer the emotional states the person is likely experiencing. Justify your reasoning with evidence where possible. You may choose multiple emotions such as: 'Sadness,Pain', 'Excitement,Confidence', or 'Fear,Disapproval'. Use only the following labels: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "What emotional states can be inferred from the person’s behavior in the video? Consider their facial expressions, tone of voice, and spoken content. Explain your reasoning if possible. Output should be a comma-separated list, such as: 'Happiness,Excitement' or 'Disconnection,Sympathy'. Use only these labels: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "From what you observe and hear in the video, deduce the emotional states conveyed by the person. Describe the reasoning process based on visual, auditory, and linguistic signals. You may list multiple emotions such as 'Anticipation,Happiness' or 'Esteem,Affection'. Choose only from the following valid labels: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "Determine the person’s emotional state in the video through multimodal reasoning—using facial expression, vocal signals, and speech content. Justify each selected emotion with evidence from the video. Examples: 'Pain,Suffering', 'Surprise,Engagement', or 'Embarrassment,Fatigue'. Choose from: [Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning].",
            "What emotional states can be logically inferred from the expressions, tone, and speech of the person in the video? Include multiple emotions if supported by the context. Provide brief reasoning where applicable. Valid labels: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.",
            "Watch the video and reason out the emotional states the individual may be experiencing. Use contextual clues across modalities. Multiple emotions are allowed if well-supported.",
            "Based on the combined visual, audio, and linguistic information, infer and list the person’s emotional states. This is a multi-label task. Support your answer with brief reasoning if possible.",
            "Please perform multimodal reasoning on the video and deduce the emotions being expressed. More than one emotion may apply. Highlight the cues that led to each conclusion.",
            "Consider the full multimodal context—how the person looks, sounds, and what they say. What emotions can be inferred? You may list more than one, with optional explanation.",
            "What emotions can be reasonably concluded from this video? Your answer may include multiple categories if justified by the multimodal evidence."
        ]
        
        self.think_instruction_pool = [
            "Valid labels include: Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning. Watch the video, explain your reasoning in <think>, and write all relevant emotional labels in <answer>.",
            "After analyzing the person’s facial expression, voice tone, and spoken words, what emotions are they experiencing? Think through your interpretation in <think>. Then in <answer>, list all applicable labels such as 'Fear,Disapproval' or 'Sadness,Yearning' from the valid label list.",
            "Use <think> to explain your interpretation based on multimodal evidence in the video. Then in <answer>, identify any emotions shown by selecting from: Esteem, Surprise, Pain, Sympathy, Confidence, Anger, Disgust, Fatigue, Disconnection, Suffering, Affection, Happiness, etc.",
            "From the way the person looks, speaks, and behaves in the video, what mix of emotions do they seem to convey? Write your reasoning in <think> and provide your answer in <answer> using any of the following labels: Anticipation, Anxiety, Sadness, Excitement, Yearning, Doubt/Confusion, and others from the valid list.",
            "What combination of emotions—such as 'Engagement,Excitement' or 'Suffering,Disconnection'—best reflects the person’s internal state? Think aloud in <think> and provide your label set in <answer>, choosing only from the full emotion taxonomy.",
            "Please observe the video and explain in <think> what emotional states are evident. Then choose all applicable labels and report them in <answer> from the following: Pain, Anger, Surprise, Sympathy, Confidence, Annoyance, Fear, Fatigue, Happiness, Sadness, Pleasure, and so on.",
            "This is a multi-label classification task. Based on the video’s multimodal information, explain in <think> how you infer the person’s emotions. Then provide your final list of labels in <answer>. Example: 'Embarrassment,Esteem'. Use only the approved label set.",
            "Think carefully about how the speaker expresses themselves through tone, gesture, and words. What emotions are reflected? Justify in <think> and select your labels in <answer>, such as 'Pleasure,Confidence' or 'Fear,Doubt/Confusion'.",
            "You are asked to predict all emotions present in the video. Begin by reasoning through facial, vocal, and semantic clues in <think>. Then select one or more from: Affection, Suffering, Surprise, Annoyance, Embarrassment, Sensitivity, and others—record them in <answer>.",
            "Describe the emotional experience conveyed in the video using multimodal understanding. Use <think> to show your reasoning, then list all relevant emotions in <answer>. Labels can include 'Disapproval,Fear', 'Peace,Esteem', or any valid multi-emotion pairings.",
        ]

        self.task_pool = ["multi-label", "reason", "think"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "multi-label":
            answer = multi_label.replace(";", ",")
            instruction_pool = self.emotion_multi_label_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
            answer = "<think>" + answer + "</think>" + "<answer>" + multi_label + "</answer>"
            instruction_pool = self.think_instruction_pool 


        if self._check_answer(answer):
            task = "multi-label"
            answer = multi_label.replace(";", ",")
            instruction_pool = self.emotion_multi_label_instruction_pool    
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")    
                               
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }
        

class CHSIMEV2SDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.sentiment_instruction_pool = [
            "Based on the combined visual, audio, and textual cues in the video, what is the speaker's overall emotional tone? Choose one label from: Positive, Negative, or Neutral.",
            "Analyze the multimodal input—facial expressions, tone of voice, and spoken words—to determine the person's emotional sentiment. Is it best described as Positive, Neutral, or Negative?",
            "What emotional attitude is conveyed by the person in the video? Consider their behavior, speech content, and vocal tone. Pick one: Neutral, Positive, or Negative.",
            "From what the speaker shows and says in the video, what is the most accurate emotional classification? Select one of the following: Negative, Neutral, Positive.",
            "Given the audio, visual, and textual signals, which category best represents the emotional state expressed: Positive, Negative, or Neutral?",
            "What sentiment is most clearly communicated by the speaker in this video clip? Choose just one label from: Neutral, Negative, Positive.",
            "Evaluate the person's emotion in this video. Based on multimodal evidence, select one sentiment label: Positive, Negative, or Neutral.",
            "Taking into account the speaker's tone, expressions, and spoken content, determine whether the emotional expression is Positive, Neutral, or Negative.",
            "From the overall mood conveyed through voice, facial behavior, and language, what sentiment is being expressed? Answer with one of the following: Negative, Positive, or Neutral.",
            "Based on all available information in the video, which of the following emotional sentiments applies to the speaker: Positive, Neutral, or Negative?",
        ]
        
        self.think_instruction_pool = [
            "Labels to choose from: Positive, Neutral, Negative. Based on all multimodal cues in the video—such as tone, expressions, and language—explain your reasoning in <think> and select the best-matching sentiment label in <answer>.",
            "What overall sentiment does the speaker convey in this video? Think about how they speak, move, and express themselves. Write your reasoning in <think> and your answer (Positive, Negative, or Neutral) in <answer>.",
            "Given the speaker’s body language, voice, and speech content, what emotional sentiment do they express? Use <think> to explain your analysis and choose one sentiment in <answer>: Positive, Negative, or Neutral.",
            "From the visual and auditory behaviors combined with the text, what sentiment is reflected in the speaker’s state? Write your analysis in <think>, then give your answer in <answer>. Choose from: Neutral, Negative, Positive.",
            "Analyze the speaker's overall tone using multimodal signals (face, voice, speech). What is their emotional attitude? Your <think> section should explain, and your <answer> must be one of: Negative, Neutral, or Positive.",
            "What emotional tone is shown in the video? Consider expression, vocal delivery, and verbal content. Justify your reasoning in <think>, then choose one label from Positive, Neutral, or Negative in <answer>.",
            "Watch and listen to the speaker. Based on how they act and what they say, what sentiment do they express? Explain step-by-step in <think> and write your final label (Negative, Positive, or Neutral) in <answer>.",
            "Based on the full context of the video, determine the person’s emotional sentiment. First reason in <think> using multimodal features, then provide one of the following in <answer>: Neutral, Positive, Negative.",
            "Think carefully about the mood conveyed by the speaker’s voice, expressions, and statements. What sentiment is it? Record your analysis in <think> and final judgment in <answer>: Positive, Negative, or Neutral.",
            "After reviewing facial expressions, voice modulation, and language use, what is the best sentiment label to describe the speaker’s emotion? Write your thoughts in <think> and your label (Neutral, Negative, or Positive) in <answer>.",
        ]

        self.reason_instruction_pool = [
            "Carefully analyze the person's expressions, voice, and language in the video—based on these multimodal cues, what sentiment do they most likely express: Positive, Neutral, or Negative?",
            "Positive, Neutral, or Negative—given the speaker’s tone, gestures, and verbal content, which sentiment do you infer from the combined evidence?",
            "Using all available information in the video, including behavior and emotional tone, reason out whether the sentiment is best described as Negative, Neutral, or Positive.",
            "Taking into account the speaker’s visible affect, speech delivery, and emotional cues, what is the most accurate sentiment category: Neutral, Positive, or Negative?",
            "Consider how the speaker talks, looks, and reacts in this clip—would you classify their emotional tone as Positive, Negative, or Neutral?",
            "Reason from the combined visual, audio, and textual context: which of the three sentiments—Positive, Neutral, Negative—best fits the speaker's emotional state?",
            "Is the speaker conveying a generally Negative, Positive, or Neutral emotion based on how they act and express themselves across modalities?",
            "Based on vocal intensity, facial behavior, and the emotional content of speech, what emotional sentiment is being conveyed: Neutral, Negative, or Positive?",
            "From the video’s multimodal signals, including facial movements, speech patterns, and tone, infer the dominant emotional sentiment—choose from: Positive, Negative, or Neutral.",
            "Judging from the emotional dynamics across modalities, which sentiment—Positive, Neutral, or Negative—is most consistent with the speaker’s behavior?",
            "What overall emotional sentiment is being communicated by the speaker, as inferred from their expressions, tone of voice, and choice of words?",
            "Observe how the speaker looks, sounds, and speaks—what general emotional attitude do you perceive based on multimodal reasoning?",
        ]
        
        self.task_pool = ["sentiment", "reason", "think"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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

        label = row.get("annotation", "unknown")
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = label
            instruction_pool = self.sentiment_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.think_instruction_pool 

        if self._check_answer(answer):
            task = "sentiment"
            answer = label
            instruction_pool = self.sentiment_instruction_pool       
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")       
                               
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }


class CMUMOSEIDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.sentiment_instruction_pool = [
            "Considering the speaker's expressions, tone, and spoken words, which sentiment—Positive, Negative, or Neutral—best captures the overall emotional atmosphere of the video?",
            "What sentiment is conveyed by the speaker when analyzing their vocal cues, facial movements, and textual content? Choose from: Neutral, Positive, or Negative.",
            "Examine the emotional state of the person based on what they say, how they say it, and how they appear. Which label fits best: Negative, Positive, or Neutral?",
            "Integrating the visual, audio, and textual modalities, how would you classify the speaker's sentiment? Select one of: Positive, Neutral, or Negative.",
            "By observing the speaker's face, listening to their tone, and interpreting their words, determine whether their sentiment is Neutral, Negative, or Positive.",
            "Assess the emotional tone presented across multiple modalities in the video. What is the most accurate sentiment: Positive, Negative, or Neutral?",
            "How does the speaker come across emotionally when combining facial cues, tone of speech, and dialogue content? Pick one: Negative, Neutral, or Positive.",
            "Based on the integrated signals from the video—including visuals, voice, and language—what sentiment is expressed? Choose one: Positive, Neutral, or Negative.",
            "What emotional stance is the speaker expressing in the clip? Use clues from all modalities to select: Neutral, Positive, or Negative.",
            "Analyze the overall emotional tone from the speaker's multimodal behavior. Which sentiment label applies best: Negative, Neutral, or Positive?",
        ]

        self.sentiment_instruction_pool_no_neutral_class = [
            "Considering the speaker's expressions, tone, and spoken words, which sentiment—Positive or Negative—best captures the overall emotional atmosphere of the video?",
            "What sentiment is conveyed by the speaker when analyzing their vocal cues, facial movements, and textual content? Choose from: Positive, or Negative.",
            "Examine the emotional state of the person based on what they say, how they say it, and how they appear. Which label fits best: Negative, Positive?",
            "Integrating the visual, audio, and textual modalities, how would you classify the speaker's sentiment? Select one of: Positive or Negative.",
            "By observing the speaker's face, listening to their tone, and interpreting their words, determine whether their sentiment is Negative, or Positive.",
            "Assess the emotional tone presented across multiple modalities in the video. What is the most accurate sentiment: Positive, Negative?",
            "How does the speaker come across emotionally when combining facial cues, tone of speech, and dialogue content? Pick one: Negative, or Positive.",
            "Based on the integrated signals from the video—including visuals, voice, and language—what sentiment is expressed? Choose one: Positive, or Negative.",
            "What emotional stance is the speaker expressing in the clip? Use clues from all modalities to select: Positive, or Negative.",
            "Analyze the overall emotional tone from the speaker's multimodal behavior. Which sentiment label applies best: Negative or Positive?",
        ]
        
        self.reason_instruction_pool = [
            "Given the speaker's facial expressions, vocal tone, and spoken words, what sentiment—Positive, Neutral, or Negative—can be reasonably inferred from the combined evidence?",
            "Positive, Negative, or Neutral—consider the emotional intent expressed across visual, vocal, and verbal cues, and identify the most appropriate sentiment label.",
            "By reasoning from how the person speaks, looks, and behaves, what is the overall sentiment reflected in the video? Choose one: Negative, Positive, or Neutral.",
            "Considering the full multimodal context—facial movements, speech tone, and language—what emotional sentiment is most likely present: Neutral, Positive, or Negative?",
            "When integrating information from the speaker's expressions, intonation, and statements, which of the three sentiments—Positive, Negative, Neutral—best describes their emotional state?",
            "Observe the speaker's multimodal behavior and infer the overall emotional tone. Is it best described as: Positive, Neutral, or Negative?",
            "Taking into account the speaker’s affective expressions, vocal delivery, and message content, determine their sentiment—Neutral, Negative, or Positive.",
            "What is the inferred sentiment expressed in this video, based on reasoning across visuals, voice dynamics, and textual meaning? Choose one: Negative, Neutral, or Positive.",
            "From the person's speech content, tone, and facial feedback, reason out whether the emotional sentiment is: Neutral, Positive, or Negative.",
            "Which sentiment—Positive, Negative, or Neutral—does the speaker's behavior most consistently suggest, when analyzed across all available modalities?",
            "What general emotional sentiment is conveyed by the speaker, based on a comprehensive analysis of their appearance, voice, and spoken content?",
            "Without being provided explicit options, reason through the multimodal signals and describe the overall emotional tone expressed by the person in the video.",
        ]

        self.think_instruction_pool = [
            "Choose from: Positive, Negative, or Neutral. Based on the speaker's expressions, vocal tone, and spoken content, explain your reasoning in <think> and provide the sentiment in <answer>.",
            "What emotional sentiment does the speaker communicate through their voice, expressions, and language? Reason step-by-step in <think> and conclude with one label—Neutral, Negative, or Positive—in <answer>.",
            "Considering how the person appears, sounds, and speaks, what sentiment is being conveyed? Use <think> to explain and write the answer (Negative, Neutral, or Positive) in <answer>.",
            "Use the visual, audio, and verbal information to assess the speaker’s emotional tone. In <think>, justify your reasoning. Then select one label in <answer>: Positive, Neutral, or Negative.",
            "Analyze how the speaker presents themselves—facial cues, tone, and dialogue. Reflect in <think> and give your final sentiment label in <answer>. Options: Neutral, Negative, Positive.",
            "The speaker’s sentiment must be one of the following: Negative, Positive, or Neutral. Consider multimodal evidence and explain your inference in <think>. Conclude with <answer>.",
            "Observe how the speaker communicates. What is the emotional tone based on their multimodal signals? Reason in <think> and write your final judgment—Positive, Negative, or Neutral—in <answer>.",
            "From integrated video signals, how would you describe the emotional tone? Use <think> to interpret the cues and select from: Neutral, Positive, or Negative in <answer>.",
            "Looking at the person’s voice quality, gestures, and message, what emotional polarity emerges? Analyze in <think> and determine if it is Positive, Neutral, or Negative in <answer>.",
            "What is the dominant sentiment shown by the speaker across all modalities? Describe the cause of your judgment in <think> and finalize with one of: Positive, Negative, or Neutral in <answer>.",
        ]


        self.task_pool = ["sentiment", "reason", "think"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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

        label = row.get("sentiment_label", "unknown")
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = label
            instruction_pool = self.sentiment_instruction_pool
            if answer != "neutral":
                if random.random() < 0.5:
                    instruction_pool = self.sentiment_instruction_pool_no_neutral_class
                else:
                    instruction_pool = self.sentiment_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.think_instruction_pool 

        if self._check_answer(answer):
            task = "sentiment"
            answer = label
            instruction_pool = self.sentiment_instruction_pool      
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")        
                               
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }


class CMUMOSIDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.sentiment_instruction_pool = [
            "Considering the speaker's expressions, tone, and spoken words, which sentiment—Positive, Negative, or Neutral—best captures the overall emotional atmosphere of the video?",
            "What sentiment is conveyed by the speaker when analyzing their vocal cues, facial movements, and textual content? Choose from: Neutral, Positive, or Negative.",
            "Examine the emotional state of the person based on what they say, how they say it, and how they appear. Which label fits best: Negative, Positive, or Neutral?",
            "Integrating the visual, audio, and textual modalities, how would you classify the speaker's sentiment? Select one of: Positive, Neutral, or Negative.",
            "By observing the speaker's face, listening to their tone, and interpreting their words, determine whether their sentiment is Neutral, Negative, or Positive.",
            "Assess the emotional tone presented across multiple modalities in the video. What is the most accurate sentiment: Positive, Negative, or Neutral?",
            "How does the speaker come across emotionally when combining facial cues, tone of speech, and dialogue content? Pick one: Negative, Neutral, or Positive.",
            "Based on the integrated signals from the video—including visuals, voice, and language—what sentiment is expressed? Choose one: Positive, Neutral, or Negative.",
            "What emotional stance is the speaker expressing in the clip? Use clues from all modalities to select: Neutral, Positive, or Negative.",
            "Analyze the overall emotional tone from the speaker's multimodal behavior. Which sentiment label applies best: Negative, Neutral, or Positive?",
        ]

        self.sentiment_instruction_pool_no_neutral_class = [
            "Considering the speaker's expressions, tone, and spoken words, which sentiment—Positive or Negative—best captures the overall emotional atmosphere of the video?",
            "What sentiment is conveyed by the speaker when analyzing their vocal cues, facial movements, and textual content? Choose from: Positive, or Negative.",
            "Examine the emotional state of the person based on what they say, how they say it, and how they appear. Which label fits best: Negative, Positive?",
            "Integrating the visual, audio, and textual modalities, how would you classify the speaker's sentiment? Select one of: Positive or Negative.",
            "By observing the speaker's face, listening to their tone, and interpreting their words, determine whether their sentiment is Negative, or Positive.",
            "Assess the emotional tone presented across multiple modalities in the video. What is the most accurate sentiment: Positive, Negative?",
            "How does the speaker come across emotionally when combining facial cues, tone of speech, and dialogue content? Pick one: Negative, or Positive.",
            "Based on the integrated signals from the video—including visuals, voice, and language—what sentiment is expressed? Choose one: Positive, or Negative.",
            "What emotional stance is the speaker expressing in the clip? Use clues from all modalities to select: Positive, or Negative.",
            "Analyze the overall emotional tone from the speaker's multimodal behavior. Which sentiment label applies best: Negative or Positive?",
        ]

        self.reason_instruction_pool = [
            "Given the speaker's facial expressions, vocal tone, and spoken words, what sentiment—Positive, Neutral, or Negative—can be reasonably inferred from the combined evidence?",
            "Positive, Negative, or Neutral—consider the emotional intent expressed across visual, vocal, and verbal cues, and identify the most appropriate sentiment label.",
            "By reasoning from how the person speaks, looks, and behaves, what is the overall sentiment reflected in the video? Choose one: Negative, Positive, or Neutral.",
            "Considering the full multimodal context—facial movements, speech tone, and language—what emotional sentiment is most likely present: Neutral, Positive, or Negative?",
            "When integrating information from the speaker's expressions, intonation, and statements, which of the three sentiments—Positive, Negative, Neutral—best describes their emotional state?",
            "Observe the speaker's multimodal behavior and infer the overall emotional tone. Is it best described as: Positive, Neutral, or Negative?",
            "Taking into account the speaker’s affective expressions, vocal delivery, and message content, determine their sentiment—Neutral, Negative, or Positive.",
            "What is the inferred sentiment expressed in this video, based on reasoning across visuals, voice dynamics, and textual meaning? Choose one: Negative, Neutral, or Positive.",
            "From the person's speech content, tone, and facial feedback, reason out whether the emotional sentiment is: Neutral, Positive, or Negative.",
            "Which sentiment—Positive, Negative, or Neutral—does the speaker's behavior most consistently suggest, when analyzed across all available modalities?",
            "What general emotional sentiment is conveyed by the speaker, based on a comprehensive analysis of their appearance, voice, and spoken content?",
            "Without being provided explicit options, reason through the multimodal signals and describe the overall emotional tone expressed by the person in the video.",
        ]

        self.think_instruction_pool = [
            "Choose from: Positive, Negative, or Neutral. Based on the speaker's expressions, vocal tone, and spoken content, explain your reasoning in <think> and provide the sentiment in <answer>.",
            "What emotional sentiment does the speaker communicate through their voice, expressions, and language? Reason step-by-step in <think> and conclude with one label—Neutral, Negative, or Positive—in <answer>.",
            "Considering how the person appears, sounds, and speaks, what sentiment is being conveyed? Use <think> to explain and write the answer (Negative, Neutral, or Positive) in <answer>.",
            "Use the visual, audio, and verbal information to assess the speaker’s emotional tone. In <think>, justify your reasoning. Then select one label in <answer>: Positive, Neutral, or Negative.",
            "Analyze how the speaker presents themselves—facial cues, tone, and dialogue. Reflect in <think> and give your final sentiment label in <answer>. Options: Neutral, Negative, Positive.",
            "The speaker’s sentiment must be one of the following: Negative, Positive, or Neutral. Consider multimodal evidence and explain your inference in <think>. Conclude with <answer>.",
            "Observe how the speaker communicates. What is the emotional tone based on their multimodal signals? Reason in <think> and write your final judgment—Positive, Negative, or Neutral—in <answer>.",
            "From integrated video signals, how would you describe the emotional tone? Use <think> to interpret the cues and select from: Neutral, Positive, or Negative in <answer>.",
            "Looking at the person’s voice quality, gestures, and message, what emotional polarity emerges? Analyze in <think> and determine if it is Positive, Neutral, or Negative in <answer>.",
            "What is the dominant sentiment shown by the speaker across all modalities? Describe the cause of your judgment in <think> and finalize with one of: Positive, Negative, or Neutral in <answer>.",
        ]        

        self.task_pool = ["sentiment", "reason", "think"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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

        label = row.get("sentiment_label", "unknown")
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = label
            instruction_pool = self.sentiment_instruction_pool
            if answer != "neutral":
                if random.random() < 0.5:
                    instruction_pool = self.sentiment_instruction_pool_no_neutral_class
                else:
                    instruction_pool = self.sentiment_instruction_pool
        elif task == "reason":
            answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
            instruction_pool = self.reason_instruction_pool
        elif task == "think":
            answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
            answer = "<think>" + str(answer) + "</think>" + "<answer>" + str(label) + "</answer>"
            instruction_pool = self.think_instruction_pool 

        if self._check_answer(answer):
            task = "sentiment"
            answer = label
            instruction_pool = self.sentiment_instruction_pool     
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")         
                               
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }


class MELDDatasetFT(MERDatasetFT):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

        self.emotion_instruction_pool = [
            "Given the visual, auditory, and spoken content in the video, what is the most appropriate emotion label to describe the speaker's state? Choose one from: joy, fear, sad, neutral, anger, surprise, or disgust.",
            "Based on the person's tone of voice, facial expressions, and language in the video, what emotion are they experiencing? Select one label from: anger, disgust, fear, joy, neutral, sad, surprise.",
            "What emotional category best fits the speaker's behavior and expression in the video? Consider all modalities and choose one: neutral, joy, surprise, anger, sad, disgust, or fear.",
            "Using all available cues in the video—visuals, speech, and audio delivery—what emotion is being communicated? Pick from: fear, anger, surprise, joy, sad, neutral, or disgust.",
            "Watch and listen to the video carefully. What is the speaker's emotional state based on their body language, voice, and speech content? Choose one: disgust, joy, neutral, anger, sad, fear, or surprise.",
            "What emotion is most clearly conveyed by the speaker in the video? Consider their tone, facial expression, and wording. Select one label: sad, surprise, anger, fear, joy, neutral, or disgust.",
            "From the combined visual, audio, and textual features in the video, what emotional state can be inferred? Your answer must be one of: neutral, disgust, surprise, sad, anger, joy, or fear.",
            "Analyze the multimodal information presented in the video. Which emotion is most strongly expressed by the person: joy, sad, surprise, anger, fear, disgust, or neutral?",
            "What is the emotional state of the speaker based on how they look, sound, and speak? Choose the most fitting label from: anger, surprise, joy, neutral, fear, sad, or disgust.",
            "Given the overall presentation in the video, including expressions, tone, and words, identify the person's emotion. Options: surprise, disgust, fear, joy, neutral, sad, or anger.",
        ]
        self.sentiment_instruction_pool = [
            "From the speaker's facial expressions, tone of voice, and spoken content, what sentiment do they most strongly express—Neutral, Positive, or Negative?",
            "Taking into account the visual cues, vocal inflections, and verbal content in the video, identify the overall sentiment: Positive, Negative, or Neutral.",
            "What is the most appropriate sentiment label for the speaker, based on their multimodal behavior in the video? Choose one: Negative, Neutral, or Positive.",
            "Observe the person's behavior, voice, and language—what sentiment is reflected through these combined channels? Pick one of: Positive, Neutral, or Negative.",
            "Using the available multimodal information, how would you interpret the speaker's emotional sentiment? Select from: Neutral, Negative, or Positive.",
            "Based on the video's combination of audio tone, facial expressions, and language, which sentiment—Positive, Neutral, or Negative—best represents the speaker's mood?",
            "Considering the interplay of voice, facial behavior, and textual expression, which sentiment label best describes the speaker's emotion: Neutral, Positive, or Negative?",
            "Evaluate the speaker's emotional sentiment as conveyed through multimodal input. Is it best categorized as: Negative, Positive, or Neutral?",
            "Looking at the person's expressions, listening to how they speak, and understanding what they say, which of the following sentiments apply: Positive, Negative, or Neutral?",
            "What sentiment does the speaker predominantly express through their multimodal communication in this clip? Choose from: Neutral, Negative, or Positive.",
        ]
        
        self.reason_sentiment_instruction_pool = [
            "Taking into account the speaker's expressions, vocal tone, and spoken words, what sentiment—Neutral, Positive, or Negative—can be inferred from the combined cues?",
            "Positive, Neutral, or Negative—which sentiment best describes the speaker’s emotional state as revealed through multimodal evidence including facial movements and speech?",
            "Analyze how the person talks, looks, and behaves across modalities—what sentiment label applies most accurately: Negative, Neutral, or Positive?",
            "From the integration of verbal content, tone of voice, and visual appearance, determine the speaker’s underlying emotional sentiment: Positive, Neutral, or Negative.",
            "What emotional tone does the speaker most clearly convey when all cues—language, tone, and expressions—are considered together? Options: Neutral, Negative, or Positive.",
            "Observe the video carefully and infer the sentiment conveyed. Use evidence from speech, voice, and appearance to choose: Positive, Neutral, or Negative.",
            "Given the speaker’s behavior, facial expression, and spoken delivery, what sentiment arises from their multimodal presentation: Negative, Positive, or Neutral?",
            "By interpreting the combination of how the speaker looks, speaks, and expresses meaning, what sentiment emerges: Neutral, Negative, or Positive?",
            "When synthesizing the multimodal information—vocal tone, facial cues, and verbal semantics—what sentiment is most evident: Neutral, Positive, or Negative?",
            "What sentiment category—Positive, Negative, or Neutral—best reflects the speaker’s emotional expression based on reasoning through all observed modalities?",
            "Reason through the speaker’s multimodal signals—how they appear, sound, and speak—and describe the emotional sentiment they seem to express.",
            "From a reasoning perspective, what overall emotional tone is conveyed by the speaker, based on cues from visual, audio, and linguistic channels?",
        ]
        
        self.reason_emotion_instruction_pool = [
            "From the speaker’s facial expressions, vocal tone, and spoken content, what emotion—joy, fear, sad, neutral, anger, surprise, or disgust—can best be inferred based on multimodal reasoning?",
            "Joy, fear, sad, neutral, anger, surprise, or disgust—which of these emotions is most likely expressed, considering all visual, auditory, and textual signals in the video?",
            "Carefully analyze how the speaker appears, sounds, and communicates. What emotional state are they conveying? Choose from: neutral, joy, surprise, anger, sad, disgust, or fear.",
            "Integrating information from the person’s voice, expressions, and speech, determine the most fitting emotion: fear, anger, surprise, joy, sad, neutral, or disgust.",
            "Based on the speaker’s body language, speech delivery, and tone, what is the most plausible emotional state? Select from: disgust, joy, neutral, anger, sad, fear, or surprise.",
            "What emotion is revealed through the speaker’s visual appearance, verbal message, and vocal behavior? Pick one: sad, surprise, anger, fear, joy, neutral, or disgust.",
            "Taking into account the full context of the video—including facial cues, prosody, and word choice—what emotion is being communicated? Choose one: neutral, disgust, surprise, sad, anger, joy, or fear.",
            "When reasoning across all modalities in the video, what emotion stands out most clearly? Your options: joy, sad, surprise, anger, fear, disgust, or neutral.",
            "Observe how the speaker looks, speaks, and sounds. Based on this multimodal analysis, which emotion is conveyed: anger, surprise, joy, neutral, fear, sad, or disgust?",
            "Given the expressions, intonation, and linguistic content in the video, what is the speaker’s most likely emotional state? Choose from: surprise, disgust, fear, joy, neutral, sad, or anger.",
            "Based on the speaker’s facial expressions, voice, and words, what emotion are they most likely experiencing? Justify your answer with multimodal cues.",
            "Observe the full emotional context of the video—how the speaker looks, sounds, and what they say—and infer their emotional state without relying on a fixed list of categories.",
        ]
        
        self.think_sentiment_instruction_pool = [
            "Choose one: Positive, Neutral, or Negative. Analyze the speaker's facial expression, tone, and words to determine their emotional sentiment. Provide your reasoning in <think> and final label in <answer>.",
            "What sentiment is conveyed by the speaker through their expressions, voice, and content? Think through your reasoning in <think>, then provide one label in <answer>: Neutral, Positive, or Negative.",
            "Considering how the speaker behaves, speaks, and what they say, describe your reasoning in <think> and conclude with one of the following in <answer>: Negative, Neutral, Positive.",
            "Based on the visual, audio, and language cues in the video, what is the person's emotional tone? Reflect in <think> and select from: Positive, Neutral, or Negative in <answer>.",
            "Evaluate the person’s multimodal communication style—facial behavior, vocal signals, spoken content—and explain your judgment in <think>. Then decide on a sentiment: Neutral, Negative, or Positive in <answer>.",
            "Given the integrated multimodal input, explain how you arrive at your sentiment judgment in <think>. Then, in <answer>, choose the best-matching label: Negative, Positive, or Neutral.",
            "What overall sentiment does the speaker express across modalities? Use <think> to reason through your interpretation, then write your answer—Positive, Negative, or Neutral—in <answer>.",
            "How would you describe the emotional tone communicated through the person's face, voice, and language? Detail your reasoning in <think> and choose from: Neutral, Negative, Positive in <answer>.",
            "From multimodal clues—intonation, gaze, and spoken phrases—what sentiment is shown? Justify in <think> and conclude with one of: Negative, Neutral, or Positive in <answer>.",
            "Analyze the speaker’s tone, expressions, and dialogue. What is the dominant emotional attitude? Use <think> to explain, then pick one from: Positive, Neutral, Negative in <answer>.",
            "Observe the speaker’s face, voice, and language in the video. What is their emotional sentiment? Use <think> to explain your reasoning and provide your final label in <answer>.",
            "Based on all visual, vocal, and verbal cues in the video, what general sentiment is being communicated? First reason in <think>, then summarize your conclusion in <answer>.",
        ]

        self.think_emotion_instruction_pool = [
            "Choose from: joy, fear, sad, neutral, anger, surprise, or disgust. Based on the person's facial expressions, voice tone, and speech, reason through your interpretation in <think> and write your final answer in <answer>.",
            "After watching and listening to the video, what emotion is the person likely expressing? Justify your decision in <think> using multimodal evidence, then select one label in <answer>: anger, disgust, fear, joy, neutral, sad, or surprise.",
            "What emotion can be inferred from the speaker’s behavior and expression? First explain your reasoning in <think> based on visuals, voice, and words, then give your answer in <answer>: neutral, joy, surprise, anger, sad, disgust, or fear.",
            "Using available multimodal cues—tone, expression, body movement, and verbal context—what emotion is being shown? Describe your thought process in <think> and select one of: fear, anger, surprise, joy, sad, neutral, or disgust in <answer>.",
            "Watch the speaker closely. What emotional state are they in? Think through it in <think>, considering body language, tone, and speech, then write one label in <answer>: disgust, joy, neutral, anger, sad, fear, or surprise.",
            "Determine the speaker's emotion from how they look, sound, and speak. In <think>, explain your logic clearly, then pick from: sad, surprise, anger, fear, joy, neutral, or disgust and enter it in <answer>.",
            "Given the visual, audio, and textual signals in the video, what emotion is being communicated? Use <think> to break down your reasoning and finalize with one label in <answer>: neutral, disgust, surprise, sad, anger, joy, or fear.",
            "What emotional state does the person most likely display? Reflect on their multimodal cues in <think> and select the best matching emotion in <answer>: joy, sad, surprise, anger, fear, disgust, or neutral.",
            "From the speaker’s appearance, tone, and verbal behavior, infer their emotion. Explain your interpretation step-by-step in <think>, then select one from: anger, surprise, joy, neutral, fear, sad, or disgust in <answer>.",
            "Considering all features shown in the video—facial movements, vocal tone, and speech—analyze the emotional state in <think> and choose your final answer in <answer>: surprise, disgust, fear, joy, neutral, sad, or anger.",
            "Watch the speaker’s expressions, vocal tone, and language. What emotion are they expressing? First explain your reasoning in <think>, then summarize your conclusion in <answer>.",
            "From the person’s overall behavior—how they look, sound, and what they say—what emotion do you believe they’re experiencing? Justify your view in <think> and give your answer in <answer>.",
        ]
        
        self.task_pool = ["emotion", "reason", "think", "sentiment"]
        # self.task_pool = ["emotion"]
        
        self.annotation = self.annotation[self.annotation['mode'].isin(['train'])]

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
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        
        # Whisper feature
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool
        elif task == "sentiment":
            answer = sentiment_label
            instruction_pool = self.sentiment_instruction_pool
        elif task == "reason":
            if random.random() < 0.5:
                answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
                instruction_pool = self.reason_sentiment_instruction_pool
            else:
                answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
                instruction_pool = self.reason_emotion_instruction_pool
        elif task == "think":
            if random.random() < 0.5:
                answer = self._safe_str(row.get("gpt-4o-mini_Sentiment", ""))
                answer = "<think>" + answer + "</think>" + "<answer>" + sentiment_label + "</answer>"
                instruction_pool = self.think_sentiment_instruction_pool 
            else:
                answer = self._safe_str(row.get("gpt-4o-mini_Emotion", ""))
                answer = "<think>" + answer + "</think>" + "<answer>" + emotion_label + "</answer>"
                instruction_pool = self.think_emotion_instruction_pool 
                

        if self._check_answer(answer):
            task = "emotion"
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool     
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")         
                               
        character_line = "The person in video says: {} ".format(text)
        instruction = "<Vid><VideoHere></Vid> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        if not isinstance(answer, str):
            answer = str(answer) if not (answer != answer) else "" 
        answer = self.text_processor(answer)
        return {
            "image": image,
            "video_features": video_features,
            "audio_features": audio_features,
            "video_id": video_id,
            "text": text,
            "instruction_input": instruction,
            "answer": answer,
        }













