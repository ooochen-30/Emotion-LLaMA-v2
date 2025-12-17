import glob
import os
import re
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


class MERDataset(Dataset):
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
            "How would you describe the speaker's emotions according to their voice tone and speaking style in the video? Mention signs of emotional intensity, emphasis, or tension.",
            "Please analyze the speaker's emotional state from the audio perspective. Focus on elements such as pitch, speed, pauses, stress, and whether there are any signs of emotional arousal or calmness.",
            "From the audio track, what vocal features (e.g., intonation, volume, rhythm) suggest the speaker's emotional condition? Is there evidence of anger, sadness, neutrality, or any other emotion?",
            "Analyze the speaker's emotional tone based on how they speak. What clues from their intonation, pace, and vocal emphasis indicate their mood or mental state?",
            "What does the way the speaker talks (e.g., pitch changes, loudness, pauses, word emphasis) reveal about their current emotional intensity or attitude?",
            "Focus on the acoustic features of the speaker's voice. How do those features contribute to interpreting their emotional experience in this moment?",
            "From the speaker's intonation and expressiveness, can you identify any signs of pleasure, joy, or enthusiasm? What acoustic features support your interpretation?",
            "What vocal characteristics in the audio suggest a positive emotional state? Pay attention to speech dynamics like pitch elevation, energetic delivery, and rhythmic emphasis.",
            "How would you describe the speaker's emotional tone based on their vocal delivery in the audio? Please consider pace, pitch, clarity, and emphasis.",
            "What can the speaker's tone and speaking style tell us about their emotional state? Focus on the overall rhythm, intonation, and vocal attitude.",
            "Analyze the speaker's vocal tone and expression. What kind of emotional stance or mood might their way of speaking suggest?",
            "From an audio perspective, what can you infer about how the speaker feels? Consider speech rate, voice quality, and tone.",
            "Please examine the speaker's vocal delivery in the clip. What emotional or attitudinal cues are suggested by their tone, volume, and emphasis?",
            "Without relying on facial expressions or text, what does the speaker's voice alone convey about their mood or emotional engagement?",
            "Based on the way the speaker talks in the video, what impressions do you get about their emotional state? Consider vocal traits such as steadiness, enthusiasm, or calmness.",
            "What kind of emotional tone is conveyed by the speaker's voice? Look at how they speak rather than what they say.",
            "What emotional impressions can be drawn from the speaker's voice in the audio clip? Please consider vocal qualities such as tone, pitch, intonation, rhythm, and pauses.",
            "Based solely on the audio, how would you describe the speaker's emotional expression? Analyze features like vocal stability, hesitations, energy, and emphasis.",
            "Please analyze the speaker's emotional state based on their tone and speech delivery. Are there any vocal signs of emotional discomfort, calmness, or intensity?",
            "From the audio input, what can you infer about the speaker's mood or attitude? Consider tempo, vocal fluctuations, tremors, or firmness in their voice.",
            "Without relying on the spoken content, what does the speaker's vocal delivery suggest about their emotional stance? Focus on prosodic and expressive vocal traits.",
            "How does the speaker's way of speaking—such as their tone, rhythm, and voice modulation—reflect their internal emotional condition?",
            "Based on the audio characteristics alone, what type of emotional atmosphere is conveyed by the speaker? Examine pitch range, stress, and fluency in their speech.",
            "Can you interpret the speaker's emotional engagement or discomfort using only the vocal audio? Pay attention to subtle features like pauses, breathiness, or vocal energy.",
            "Listen to how the speaker delivers their speech. What emotions, if any, are implied by the manner in which they speak rather than the meaning of the words?",
            "Evaluate the emotional content conveyed through the speaker's tone and intonation. Does their delivery suggest neutrality, excitement, tension, or another feeling?",
            "Imagine you cannot understand the language spoken—just listen to how the speaker sounds. What emotions or attitudes might be perceived through their voice alone?",
            "If you were to describe the speaker's voice as a musical performance, what emotional tone would it convey—calm, intense, uncertain, joyful, or something else? Explain why.",
            "Consider the overall delivery of the speaker—does it sound emotionally charged, emotionally flat, or somewhere in between? Justify your answer based on vocal behavior.",
        ]
        self.video_instruction_pool = [
            "Based on the visual input, what can be inferred about the person's emotional state? Please consider facial expressions, posture, and body language.",
            "What emotional impressions can be drawn from the person's appearance and behavior in the video? Focus on gestures, eye contact, facial cues, and movement.",
            "Analyze the person's body language and facial expression. What kind of emotional atmosphere might they be projecting?",
            "Without relying on audio or dialogue, what does the person's facial expression and physical behavior suggest about their mood or emotional engagement?",
            "How would you describe the person's emotional tone based on their visual behavior, such as head movements, hand gestures, or facial tension?",
            "Observe the person in the video. What emotional cues are visible from their posture, expression, or interaction with the environment?",
            "From a visual perspective, what non-verbal signals—such as facial muscle movement, body orientation, or hand positioning—reveal about the person's emotional condition?",
            "What does the individual's overall appearance and motion in the video convey in terms of their emotional state? Focus on observable behaviors rather than assumptions.",
            "How do the person's facial dynamics and physical gestures contribute to interpreting their emotional experience at that moment?",
            "What kind of mood or emotional stance is visually communicated by the person through their expressions and body language?",  
            "What can be inferred about the person's emotional state based on their facial expression and posture in the image or video?",
            "Please describe any observable emotional signals in the video. Focus on the person's eye gaze, mouth movement, and body orientation.",
            "How does the individual's body language and facial behavior contribute to understanding how they might be feeling in this moment?",
            "From a purely visual perspective, what kind of emotional cues can be observed in the person's expression and behavior?",
            "Does the person appear emotionally engaged, tense, or indifferent based on their posture and expression? Please explain your reasoning using visual cues.",
            "If you had to guess the person's mood from their visual appearance alone, what would you say? Consider facial muscles, eye openness, and overall physical demeanor.",
            "How might the position of the hands, head tilt, and eye direction help indicate the emotional state of the individual in the frame?",
            "Are there any signs of emotional subtlety in the person's facial features or movements? For instance, slight frowning, squinting, or micro-expressions?",
            "What does the visual scene suggest about how the person might be feeling? Consider their physical stance and whether their body appears relaxed or tense.",
            "Without making assumptions about the context, what visible expressions or gestures might indicate how the person is feeling internally?",
            "What changes in the person's posture or facial expression can be observed across the video? Do these changes suggest any shifts in emotional state?",
            "How does the person's interaction—or lack thereof—with their surroundings reflect their internal emotional state?",
            "What can be interpreted from the alignment between the person's facial expression and their bodily movement? Are they congruent or contrasting?",
            "Does the person's behavior in the video seem deliberate or spontaneous? What could this imply about their emotional state?",
            "Are there any noticeable physical reactions—such as blinking rate, eye darting, or subtle flinching—that might indicate emotional discomfort or surprise?",
            "What non-verbal expressions in the video could be interpreted as signs of inner conflict or uncertainty?",
            "Watch the video and describe what emotional state the person seems to be experiencing. What features led you to this conclusion?",
            "From the video clip, infer how the person might be feeling and explain your reasoning based on what you observe.",
            "Please interpret the person's emotional state in the video and explain which elements (e.g., expression, movement) support your interpretation.",
            "Based on the video content, what is your best guess about the person's emotions? Describe the cues that informed your reasoning.",
            "Look at the individual in the video. What emotions do you think they are experiencing? Please explain your judgment using observable signals.",
        ]
        self.image_instruction_pool = [
            "What emotional state might the person in the image be experiencing? Please support your answer using visual cues such as facial expression, body posture, and gaze direction.",
            "Based on the image, what can be inferred about the emotional interaction between the individuals present? Consider facial expressions, eye contact, and spatial relationships.",
            "Please analyze the person's demeanor in the picture. What might their emotional response be in this moment, and what visual details support your reasoning?",
            "How would you interpret the emotional tone of the scene in the image? Focus on expressions, physical behavior, and any contextual elements present in the background.",
            "Look closely at the person's face and posture in the image. What kind of emotional experience might they be having, and what aspects of the image inform your judgment?",
            "Are there any clues in the person's body language, expression, or surroundings that reveal how they may be feeling?",
            "Observe the individual in the image. Without assuming any specific emotion, what impressions do you get about their mood based on how they are presented?",
            "What emotional atmosphere is conveyed by this image? Consider how the person's expression, the environment, and any visible interactions contribute to this impression.",
            "Based on the image alone, how might you describe the social or emotional dynamic occurring in this moment?",
            "What can be inferred about the person's emotional state based on their facial expression, posture, and overall demeanor in the image?",
            "Please analyze the visual cues in the image—such as the person's expression, gaze direction, and physical stance—to describe what they might be feeling.",
            "How do the person's appearance and body language in the image contribute to your understanding of their current emotional experience?",
            "What visual elements in the image suggest how the person might be reacting to something emotionally? Consider facial features, hand position, and setting.",
            "Observe the person's expression and physical posture. What emotions might be present in this moment, based on what you see?",
            "What mood or emotional tone is conveyed through the person's visual appearance in the image? Focus on what is observable rather than assumed.",
            "Based solely on the visual information in the image, how would you describe the emotional atmosphere surrounding the individual?",
            "Are there any subtle details in the image—like eye contact, lip tension, or body orientation—that offer clues about the person's inner state?",
            "What kind of emotional moment does the image seem to capture? Describe what stands out in the person's appearance or environment that supports your answer.",
            "How might the combination of the person's expression and the surrounding context help you infer their emotional reaction to an unseen event?",
            "As a professional in emotional reasoning, interpret the person's emotional state in the image. Please explain your reasoning process based on observable details.",
            "Imagine you are an emotion inference specialist. What can you deduce about the individual's mood from this image? Support your answer with evidence from their appearance and environment.",
            "You specialize in analyzing emotions from images. Carefully examine the person's expression and body language—what emotion might they be experiencing?",
            "You are an expert in non-verbal emotional cues. Using the image provided, infer the person's likely emotional state and describe the visual indicators that support your conclusion.",
            "As someone trained in reading emotional states from static images, please describe what the person might be feeling and why.",
            "You are a visual emotion analyst. Based on the details in this image, what kind of emotional state is being communicated, and how do the facial and contextual elements contribute?",
            "Take on the role of an emotion psychologist. What can you infer about the person's inner state in this picture? Walk through the thought process behind your analysis.",
            "You are a specialist in understanding subtle emotions through visual observation. Please interpret the image and explain the clues that led you to your conclusion.",
            "As a trained emotional behavior analyst, infer the emotional tone of the individual in the image. Use body position, expression, and setting to support your reasoning.",
            "You are an expert in emotion recognition. Please analyze the emotional state of the person in the image using visual cues such as facial expression, posture, and surrounding context.",
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

    def string_to_list(self, str):

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


    def _check_answer(self, answer):
        change_flag = False
        if not answer or not isinstance(answer, str):
            change_flag = True
        elif len(answer.strip().split()) > 120:
            change_flag = True
        return change_flag

    
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

class MER2023Dataset(MERDataset):
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

        self.task_pool = ["emotion", "audio", "image"]
        
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
            
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")
        
        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "emotion"
            answer = label
            instruction_pool = self.emotion_instruction_pool            
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                  
        character_line = "The person in video says: {}. ".format(text)
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


class CAERDataset(MERDataset):
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
        self.task_pool = ["emotion", "audio", "image"]

        
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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 
            
        if self._check_answer(answer):
            task = "emotion"
            answer = label
            instruction_pool = self.emotion_instruction_pool 
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                        
        character_line = "The person in video says: {}. ".format(text)
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

        
class DFEWDataset(MERDataset):
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

        self.task_pool = ["emotion", "audio", "image"]
        
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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 
            
        if self._check_answer(answer):
            task = "emotion"
            answer = label
            instruction_pool = self.emotion_instruction_pool    
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                                
        character_line = "The person in video says: {}. ".format(text)
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


class MCEIUDataset(MERDataset):
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
        self.audio_instruction_pool = [
            "Based on the speaker's tone, pitch, and rhythm in the audio, what emotions or intentions can you infer? Feel free to describe both emotional states and communicative goals.",
            "How would you describe the speaker's emotional and intentional state based on their vocal tone and speaking style? Consider emotional intensity, emphasis, and signs of engagement, agreement, or questioning.",
            "Please analyze the speaker's vocal delivery to infer both their emotion and intent. Focus on pitch, speed, pauses, stress, and any cues of arousal, calmness, or communicative purpose.",
            "From the audio track, what vocal patterns (such as intonation, volume, and rhythm) reveal how the speaker is feeling—and what they might be trying to achieve or express?",
            "Analyze the speaker's voice in terms of both emotional tone and communicative function. What clues in their pace, emphasis, or tone suggest their mood and intent?",
            "What does the way the speaker talks—through pitch shifts, vocal intensity, and phrasing—reveal about both their emotional state and possible intention (e.g., suggesting, consoling, questioning)?",
            "Focus on the acoustic features of the speaker's voice and infer both the emotional experience and communicative intent being conveyed. What does their tone suggest they're feeling or trying to do?",
            "Can you interpret the speaker's emotional engagement or discomfort using only the vocal audio? Pay attention to subtle features like pauses, breathiness, or vocal energy.",
            "Listen to how the speaker delivers their speech. What emotions, if any, are implied by the manner in which they speak rather than the meaning of the words?",
            "Evaluate the emotional content conveyed through the speaker's tone and intonation. Does their delivery suggest neutrality, excitement, tension, or another feeling?",
        ]
        self.video_instruction_pool = [
            "Based on the visual input, including the person's facial expressions, gestures, and interactions with the environment, what can be inferred about their emotional state and possible intention?",
            "What emotional and intentional signals can be observed through the person's appearance, actions, and surrounding context in the video? Pay attention to posture, eye contact, and physical movements.",
            "Analyze the person's visual behavior—such as facial expression, body orientation, or object interaction. What emotional state and communicative intent might they be conveying?",
            "Without relying on audio, what does the person's facial dynamics, physical actions, and environmental engagement suggest about how they feel and what they are trying to do or express?",
            "How would you interpret the person's emotional tone and potential intent, based on their head movements, hand gestures, facial expressions, and how they interact with their surroundings?",
            "Observe the individual in the video. What visible emotional cues and goal-directed behaviors (e.g., approaching, rejecting, offering) can be identified from their body language and context?",
            "From a visual standpoint, what non-verbal behaviors—like facial tension, posture shifts, or object use—suggest both the person's emotional condition and communicative purpose?",
            "What does the person's overall presence and motion in the video convey in terms of their emotional state and intended action? Focus on observable, context-grounded behaviors.",
            "How do changes in the person's facial expression, body movement, and spatial orientation contribute to interpreting their emotional experience and intention in the moment?",
            "What mood and intention does the person appear to communicate through their visual behavior, including expressions, gestures, and interactions with other elements in the scene?",
        ]
        self.image_instruction_pool = [
            "What emotional state and possible intention might the person in the image be experiencing? Support your reasoning using visual cues such as facial expression, body posture, and gaze direction.",
            "Based on the visual details in the image, what can you infer about both the emotional connection and communicative intent between the individuals shown? Consider facial expressions, eye contact, and body positioning.",
            "Please analyze the person's demeanor in the image. What emotions and underlying motivations might be present, and what specific visual elements support your interpretation?",
            "How would you interpret the emotional tone and potential goal of the person or scene in the image? Focus on visible expressions, body language, and contextual cues in the environment.",
            "Looking closely at the person's face and body posture, what kind of emotional experience and behavioral intention might they be expressing? Justify your answer using what you observe in the image.",
            "Are there any indicators in the person's body language, facial expression, or environment that suggest both how they feel and what they might be trying to communicate or achieve?",
            "Observe the individual in the image. Without assuming a specific emotion, what can you infer about their emotional state and purpose based on how they are visually presented?",
            "What kind of emotional atmosphere and potential intent does the image convey? Consider facial cues, gestures, proximity to objects or people, and any visual tension in the scene.",
            "Based solely on the image, how would you describe the emotional and social dynamic at play? What behaviors, expressions, or spatial relationships guide your interpretation?",
            "From the person's expression, posture, and the overall visual composition, what can be inferred about both their emotional condition and any intention or response they may be directing toward others or the environment?",
        ]
        
        self.task_pool = ["emotion", "audio", "image", "intent"]
        
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
        
        # whisper-large-v3
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
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "intent"
            answer = intent_label
            instruction_pool = self.intent_instruction_pool     
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                               
        character_line = "The person in video says: {}. ".format(text)
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


class E3Dataset(MERDataset):
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
        self.task_pool = ["emotion", "audio", "image"]
        
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

        label = row.get("emotion", "unknown")
        text = row.get("conversation", "")
        if not isinstance(text, str):
            text = str(text) if not (text != text) else "" 
        person = row.get("person", "")      

        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            instruction_pool = self.emotion_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

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


class IEMOCAPDataset(MERDataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_path=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_path)

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
        self.emotion_instruction_pool = [ # Anger, Happiness, Disgust, Surprise, Fear, Sadness, Excited, Neutral, Frustration
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
        self.task_pool = ["emotion", "audio", "image"]
        
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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "emotion":
            answer = label
            if answer in ['Anger', 'Sadness', 'Neutral', 'Happiness']:
                instruction_pool = self.emotion_4_class_instruction_pool
            else:
                instruction_pool = self.emotion_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "emotion"
            answer = label
            if answer in ['Anger', 'Neutral', 'Sadness', 'Happiness']:
                instruction_pool = self.emotion_4_class_instruction_pool
            else:
                instruction_pool = self.emotion_instruction_pool    
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                                           
        character_line = "The person in video says: {}. ".format(text)
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
        
       
class MAFWDataset(MERDataset):
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
        self.task_pool = ["emotion", "audio", "image", "multi-label"]
   
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
        
        # whisper-large-v3
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
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 
            
        if self._check_answer(answer):
            task = "emotion"
            answer = single_label
            instruction_pool = self.emotion_instruction_pool 
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")  
                                       
        character_line = "The person in video says: {}. ".format(text)
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

         
class BOLDDataset(MERDataset):
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
        self.task_pool = ["multi-label", "audio", "image"]
        
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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "multi-label":
            answer = multi_label.replace(";", ",")
            instruction_pool = self.emotion_multi_label_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "multi-label"
            answer = multi_label.replace(";", ",")
            instruction_pool = self.emotion_multi_label_instruction_pool    
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")    
                               
        character_line = "The person in video says: {}. ".format(text)
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
        

class CHSIMEV2SDataset(MERDataset):
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

        self.task_pool = ["sentiment", "audio", "image"]
        
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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = label
            instruction_pool = self.sentiment_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "sentiment"
            answer = label
            instruction_pool = self.sentiment_instruction_pool       
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")       
                               
        character_line = "The person in video says: {}. ".format(text)
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


class CMUMOSEISDataset(MERDataset):
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

        self.task_pool = ["sentiment", "audio", "image"]

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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = label
            instruction_pool = self.sentiment_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "sentiment"
            answer = label
            instruction_pool = self.sentiment_instruction_pool      
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")        
                               
        character_line = "The person in video says: {}. ".format(text)
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


class CMUMOSISDataset(MERDataset):
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

        self.task_pool = ["sentiment", "audio", "image"]
        
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
        
        # whisper-large-v3
        audio_features = self._load_npy(video_id, "whisper-large-v3")
        
        # EVA-ViT-G
        video_features = self._load_npy(video_id, "EVA-ViT-G")

        # random task
        task = random.choice(self.task_pool)
        if task == "sentiment":
            answer = label
            instruction_pool = self.sentiment_instruction_pool
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "sentiment"
            answer = label
            instruction_pool = self.sentiment_instruction_pool     
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")         
                               
        character_line = "The person in video says: {}. ".format(text)
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


class MELDDataset(MERDataset):
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
        
        self.task_pool = ["emotion", "audio", "image", "sentiment"]
        
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
        
        # whisper-large-v3
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
        elif task == "audio":
            audio_response_pool = ["response_AudioReasoner", "response_Qwen2Audio"]
            answer = row.get(random.choice(audio_response_pool), "")
            instruction_pool = self.audio_instruction_pool
            video_features = torch.zeros(1, 64, 1408)
            image = torch.zeros_like(image)
        elif task == "image":
            answer = row.get("response_Qwen25vl_72B", "")
            instruction_pool = self.image_instruction_pool 
            video_features = torch.zeros(1, 64, 1408)
            audio_features = torch.zeros(1, 64, 1280) 

        if self._check_answer(answer):
            task = "emotion"
            answer = emotion_label
            instruction_pool = self.emotion_instruction_pool     
            audio_features = self._load_npy(video_id, "whisper-large-v3")
            video_features = self._load_npy(video_id, "EVA-ViT-G")         
                               
        character_line = "The person in video says: {}. ".format(text)
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













