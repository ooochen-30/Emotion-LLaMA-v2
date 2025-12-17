import numpy as np
import my_utils.config as config
import av
import os
import cv2
import glob
import random
from PIL import Image
from tqdm import tqdm

import pandas as pd
import timm.models.hub as timm_hub
from timm.models import create_model

import torch
import torchaudio
import torch.nn.functional as F
from torchvision import transforms
import soundfile as sf
from moviepy.editor import VideoFileClip

from transformers import (
    AutoFeatureExtractor, 
    WhisperModel
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def extract_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def extract_audio_from_video(video_path):
    try:
        video = VideoFileClip(video_path)
        if video.audio is None: return None, None
        audio_array = video.audio.to_soundarray(fps=16000)
        if audio_array.ndim > 1: audio_array = audio_array.mean(axis=1)
        return audio_array, 16000
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, None 


def read_video_pyav(container, indices, clip_len=16):
    frames = []
    container.seek(0)
    start_index, end_index = indices[0], indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index: break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) == 0: raise ValueError()
    while len(frames) < clip_len:
        frames.append(frames[-1].copy())
    return np.stack(frames)

def read_video_cv2(file_path, indices, clip_len=16):
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or total_frames <= 0:
        raise ValueError()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if clip_len is not None and len(frames) < clip_len:
        if len(frames) == 0:
            raise ValueError()
        last_frame = frames[-1]
        while len(frames) < clip_len:
            frames.append(last_frame.copy())

    return frames 


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len <= converted_len:
        return np.clip(np.linspace(0, seg_len - 1, num=clip_len), 0, seg_len - 1).astype(np.int64)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    return np.clip(np.linspace(start_idx, end_idx, num=clip_len), start_idx, end_idx - 1).astype(np.int64)

def adaptive_downsample_with_padding(features, target_len=128):
    if features.shape[1] < target_len:
        features = F.pad(features, (0, 0, 0, target_len - features.shape[1]))
    features = features.transpose(1, 2)
    features = F.adaptive_avg_pool1d(features, target_len)
    return features.transpose(1, 2)

def spatiotemporal_downsample(x, target_h, target_w, target_t):
    B, T, H, W, C = x.shape

    x = x.permute(0, 4, 1, 2, 3)

    x = F.adaptive_avg_pool3d(x, (T, target_h, target_w)) 

    if T != target_t:
        x = F.adaptive_avg_pool3d(x, (target_t, target_h, target_w))

    x = x.permute(0, 2, 3, 4, 1) 

    B, T, H, W, C = x.shape
    out = x.reshape(B, T * H * W, C)

    return out



class FeatureExtractor:

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"FeatureExtractor is initialized on device: '{self.device}'")
        
        self._whisper_model = None
        self._whisper_feature_extractor = None
        self._eva_model = None
        self._eva_transform = None

    def _load_whisper_if_needed(self):
        if self._whisper_model is None:
            model_path = "/home/user/big_space/models/openai/whisper-large-v3"
            self._whisper_model = WhisperModel.from_pretrained(model_path).to(self.device)
            self._whisper_model.eval()
            self._whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            # print("Whisper model loaded.")

    def _load_eva_if_needed(self):
        if self._eva_model is None:
            self._eva_model = create_model('eva_giant_patch14_224', pretrained=False, num_classes=0, global_pool='')
            cached_file = '/home/user/.cache/torch/hub/checkpoints/eva_vit_g.pth'
            if not os.path.exists(cached_file):
                url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
                cached_file = timm_hub.download_cached_file(url, check_hash=False, progress=True)
            
            state_dict = torch.load(cached_file, map_location="cpu")
            if 'model' in state_dict: state_dict = state_dict['model']
            self._eva_model.load_state_dict(state_dict, strict=False)
            self._eva_model = self._eva_model.to(self.device)
            self._eva_model.eval()

            self._eva_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # print("EVA-ViT-G model loaded.")


    def _extract_single_whisper_feature(self, file_path, save_path=None):
        self._load_whisper_if_needed()
        try:
            audio_data_for_extractor = None
            current_sr = 16000 

            if not file_path.endswith('.wav'):
                waveform_np, sr = extract_audio_from_video(file_path)
                if waveform_np is None: raise ValueError()
                audio_data_for_extractor, current_sr = waveform_np, sr
            else:
                waveform_tensor, sr = torchaudio.load(file_path)
                current_sr = sr
                if waveform_tensor.shape[0] > 1: waveform_tensor = waveform_tensor.mean(dim=0, keepdim=True)
                if current_sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=current_sr, new_freq=16000)
                    waveform_tensor = resampler(waveform_tensor)
                audio_data_for_extractor = waveform_tensor.squeeze().numpy()

            if audio_data_for_extractor is None: raise ValueError()

            inputs = self._whisper_feature_extractor(audio_data_for_extractor, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            decoder_input_ids = torch.tensor([[self._whisper_model.config.decoder_start_token_id]], device=self.device)

            with torch.no_grad():
                output = self._whisper_model(input_features=input_features, decoder_input_ids=decoder_input_ids)

            encoder_last_hidden_state = output.encoder_last_hidden_state
            audio_feature = adaptive_downsample_with_padding(encoder_last_hidden_state, target_len=64)

            if save_path:
                np.save(save_path, audio_feature.cpu().numpy())
            return audio_feature
        except Exception as e:
            print(f"fail: {file_path}, error: {e}")
            return None

    def _extract_single_eva_feature(self, file_path, save_path=None):
        self._load_eva_if_needed()
        try:
            cap = cv2.VideoCapture(file_path)
            seg_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=seg_len)

            try:
                video = read_video_cv2(file_path, indices, clip_len=16)
            except Exception:
                container = av.open(file_path)
                seg_len = container.streams.video[0].frames
                indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=seg_len)
                video = read_video_pyav(container, indices, clip_len=16)

            images_tensor = torch.stack([self._eva_transform(Image.fromarray(f)) for f in video], dim=0).to(self.device) 

            with torch.no_grad():
                feature = self._eva_model.forward_features(images_tensor)

            feature = feature[:, 1:, :].reshape(1,16,16,16,-1)
            feature = spatiotemporal_downsample(feature, 2, 2, 16) 

            if save_path:
                np.save(save_path, feature.cpu().numpy())

            return feature
        except Exception as e:
            print(f"fail:{file_path}, error: {e}")
            return None

    def extract_whisper_audio_features(self, dataset_name):
        if os.path.isfile(dataset_name):
            # print(f"Processing single audio file: {dataset_name}")
            return self._extract_single_whisper_feature(dataset_name)

        audio_root = config.PATH_TO_RAW_AUDIO[dataset_name]
        save_path = os.path.join(config.PATH_TO_FEATURES[dataset_name], "whisper-large-v3")

        
        os.makedirs(save_path, exist_ok=True)
        # print(f"Processing audio files from: {audio_root}")

        for file_path in tqdm(glob.glob(os.path.join(audio_root, '*.wav'))):
            filename = os.path.splitext(os.path.basename(file_path))[0]
            save_file = os.path.join(save_path, filename + '.npy')
            self._extract_single_whisper_feature(file_path, save_file)
        # print("Audio feature extraction complete.")

    def extract_eva_vit_g_features(self, dataset_name):
        if os.path.isfile(dataset_name):
            # print(f"Processing single video file: {dataset_name}")
            return self._extract_single_eva_feature(dataset_name)

        video_root = config.PATH_TO_RAW_VIDEO[dataset_name]
        save_path = os.path.join(config.PATH_TO_FEATURES[dataset_name], "EVA-ViT-G")

        os.makedirs(save_path, exist_ok=True)
        # print(f"Processing video files from: {video_root}")

        video_files = []
        for ext in ['*.mp4', '*.mkv', '*.avi']:
            video_files.extend(glob.glob(os.path.join(video_root, ext)))

        for file_path in tqdm(video_files):
            filename = os.path.splitext(os.path.basename(file_path))[0]
            save_file = os.path.join(save_path, filename + '.npy')
            self._extract_single_eva_feature(file_path, save_file)
        # print("Video feature extraction complete.")


if __name__ == '__main__':
    import fire
    fire.Fire(FeatureExtractor)