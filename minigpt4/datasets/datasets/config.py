# config_datasets.py

import pandas as pd
import os

def load_dataset_config(dataset_name, model=['model']):
    config_map = {
        "BOLD":         lambda: load_bold(model),
        "CAER":         lambda: load_caer(model),
        "CH-SIMS_v2_s": lambda: load_ch_sims(model),
        "CMU-MOSEI":    lambda: load_mosei(model),
        "CMU-MOSI":     lambda: load_mosi(model),
        "DFEW":         lambda: load_dfew(model),
        "E3":           lambda: load_e3(model),
        "IEMOCAP":      lambda: load_iemocap(model),
        "IEMOCAP_Emotion4":      lambda: load_iemocap(model),
        "MAFW":         lambda: load_mafw(model),
        "MAFW_single_label":     lambda: load_mafw(model),
        "MAFW_multi_label":      lambda: load_mafw_multi(model),
        "MC-EIU_emotion": lambda: load_mc_eiu_emotion(model),
        "MC-EIU_intent":  lambda: load_mc_eiu_intent(model),
        "MELD_emotion":         lambda: load_meld_emotion(model),
        "MELD_sentiment":       lambda: load_meld_sentiment(model),
        "MER2023":      lambda: load_mer2023(model),
    }

    if dataset_name not in config_map:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return config_map[dataset_name]()

# BOLD
def load_bold(model=['model']):
    input_file = "/home/user/big_space/datasets/BOLD/BOLD_label.csv"
    video_dir = "/home/user/big_space/datasets/BOLD/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['val'])]

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['Emotion_label']
        text_whisper = row['text_whisper']
        mode = row['mode']
        
        qwen_omni_prompt = f"""The person in the video says: {text_whisper}. Based on the multimodal input, please identify one or more emotion categories expressed by the person in the video, separated by commas."""
        qwen_omni_system = (
                    "You are a helpful assistant specialized in multimodal emotion recognition. "
                    "Given the input video and audio, your task is to identify the emotion(s) expressed by the person or people in the video. "
                    "Your output must be one or more emotion labels strictly chosen from the following list:\n"
                    "Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, "
                    "Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, "
                    "Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.\n"
                    "Please ensure that all predicted labels come only from this list."
                    "Your output can be any combination of valid emotion labels, such as: 'Anticipation,Happiness,Excitement', 'Aversion,Annoyance,Anger', 'Engagement,Doubt/Confusion', or simply 'Disquietment'."
        )

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion(s) expressed by the person or people in the video. "
            "Your output must be one or more emotion labels strictly chosen from the following list:\n"
            "Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, "
            "Disquietment, Doubt/Confusion, Embarrassment, Engagement, Esteem, Excitement, Fatigue, Fear, "
            "Happiness, Pain, Peace, Pleasure, Sadness, Sensitivity, Suffering, Surprise, Sympathy, Yearning.\n"
            "Please ensure that all predicted labels come only from this list."
            "Your output can be any combination of valid emotion labels, such as: 'Anticipation,Happiness,Excitement', 'Aversion,Annoyance,Anger', 'Engagement,Doubt/Confusion', or simply 'Disquietment'."
            "The person in the video says: {text_whisper}. Based on the multimodal input, please identify one or more emotion categories expressed by the person in the video, separated by commas."
        ).format(text_whisper=text_whisper)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result        
        
        
# CAER
def load_caer(model=['model']):
    input_file = "/home/user/big_space/datasets/CAER/CAER_label.csv"
    video_dir = "/home/user/big_space/datasets/CAER/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'validation'])]

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        label = row['emotion_label']
        text_whisper = row['text_whisper']
        mode = row['mode']
        
        qwen_omni_prompt = f"""The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, fear, happy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
        )

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, fear, happy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text_whisper=text_whisper)

        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result        
        
        
# CH-SIMS_v2_s
def load_ch_sims(model=['model']):
    input_file = "/home/user/big_space/datasets/CH-SIMS_v2_s/CH-SIMS_v2_s_label.csv"
    video_dir = "/home/user/big_space/datasets/CH-SIMS_v2_s/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'valid'])]

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['annotation']
        text = row['text']
        mode = row['mode']
        
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis. "
            "Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text=text)
        
        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis. "
            "Given the input video, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."
        ).format(text=text)

        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path,
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system

        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
            
        result.append(item)
    return result        
        
        
# DEFW
def load_dfew(model=['model']):
    input_file = "/home/user/big_space/datasets/DFEW/DFEW_label.csv"
    video_dir = "/home/user/big_space/datasets/DFEW/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test'])]

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['label']
        text_whisper = row['sentence']
        mode = row['mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "angry, disgust, fear, happy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text_whisper=text_whisper)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "angry, disgust, fear, happy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text_whisper=text_whisper)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# CMU-MOSEI
def load_mosei(model=['model']):
    input_file = "/home/user/big_space/datasets/CMU-MOSEI/CMU-MOSEI_label.csv"
    video_dir = "/home/user/big_space/datasets/CMU-MOSEI/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'valid'])]

    result = []

    for _, row in df.iterrows():
        video_id = row['video_id']+"_"+str(row['clip_id'])
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        text = row['text']
        mode = row['mode']
        label = row['sentiment_label']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis. "
            "Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."
        )
        
        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis. "
            "Given the input video, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."
        ).format(text=text)

        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path,
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system

        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
            
        result.append(item)
    return result


# CMU-MOSI
def load_mosi(model=['model']):
    input_file = "/home/user/big_space/datasets/CMU-MOSI/CMU-MOSI_label.csv"
    video_dir = "/home/user/big_space/datasets/CMU-MOSI/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'valid'])]

    result = []

    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        text = row['text']
        mode = row['mode']
        sentiment_label = row["sentiment_label"]

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis."
            "Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."
        )
        
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis."
            "Given the input video, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."            
            "The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."
        ).format(text=text)
        qwen25_vl_system = ""

        item = {
            "video_id": video_id,
            "mode": mode,
            "label": sentiment_label,
            "video_path": video_path,
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system

        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system

        result.append(item)

    return result


# E3
def load_e3(model=['model']):
    input_file = "/home/user/big_space/datasets/E3/E3_label.csv"
    video_dir = "/home/user/big_space/datasets/E3/video"
    df = pd.read_csv(input_file)
    df = df[df['set'].isin(['val', 'test'])]

    result = []

    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['emotion']
        text = row['conversation']
        mode = row['set']
        person = row["person"]

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by {person} in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition."
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "angry, disgust, happy, sad, sarcastic, scared, shy, surprised.\n"
            "Please ensure that all predicted labels come only from this list."
        )

        gpt_prompt = f""""""
        gpt_system = (
            "You are a helpful assistant for emotion classification. "
        )
        
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, happy, sad, sarcastic, scared, shy, surprised.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by {person} in the video."
        ).format(text=text, person=person)
        qwen25_vl_system = ""

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system

        if 'gpt' in model:
            item["prompt"] = gpt_prompt
            item["system"] = gpt_system
            
        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system

        result.append(item)

    return result


# IEMOCAP
def load_iemocap(model=['model']):
    input_file = "/home/user/big_space/datasets/IEMOCAP/IEMOCAP_label.csv"
    video_dir = "/home/user/big_space/datasets/IEMOCAP/video"
    df = pd.read_csv(input_file)
    df = df[df['session'].isin(['Session5'])]
    target_emotions = ['Anger', 'Neutral', 'Sadness', 'Happiness', 'Excited']
    df = df[df['official_label'].isin(target_emotions)]
    print("🎯 每个情绪类别的样本数量：")
    print(df['official_label'].value_counts())

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['official_label']
        if label == 'Excited':
            label = 'Happiness'
        text = row['text']
        mode = row['session']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "Anger, Neutral, Sadness, Happiness.\n"
            "Please ensure that all predicted labels come only from this list."
        )

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "Anger, Neutral, Sadness, Happiness.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text=text)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# MAFW
def load_mafw(model=['model']):
    input_file = "/home/user/big_space/datasets/MAFW/MAFW_label.csv"
    video_dir = "/home/user/big_space/datasets/MAFW/video"
    df = pd.read_csv(input_file, dtype={"video_id": str})
    df = df[df['mode'].isin(['test'])]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")
    
    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['single_label']
        text_whisper = row['text_whisper']
        mode = row['mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text_whisper=text_whisper)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text_whisper=text_whisper)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# MAFW Multi Label
def load_mafw_multi(model=['model']):
    input_file = "/home/user/big_space/datasets/MAFW/MAFW_label.csv"
    video_dir = "/home/user/big_space/datasets/MAFW/video"
    df = pd.read_csv(input_file, dtype={"video_id": str})
    df = df[
        (df['mode'] == 'test') &
        (df['multi_label'].notna())  # 或用 ~df['multi_label'].isna()
    ]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")
    
    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['multi_label']
        text_whisper = row['text_whisper']
        mode = row['mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text_whisper}. Based on the multimodal input, please identify one or more emotion categories expressed by the person in the video, separated by commas."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be one or more emotion labels strictly chosen from the following list:\n"
            "anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "Your output can be any combination of valid emotion labels, such as: 'disgust,contempt', 'sadness,anxiety', 'anger,surprise', 'happiness,contempt' or 'anxiety,helplessness'."
        ).format(text_whisper=text_whisper)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be one or more emotion labels strictly chosen from the following list:\n"
            "anger, anxiety, contempt, disappointment, disgust, fear, happiness, helplessness, neutral, sadness, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text_whisper}. Based on the multimodal input, please identify one or more emotion categories expressed by the person in the video, separated by commas."
            "Your output can be any combination of valid emotion labels, such as: 'disgust,contempt', 'sadness,anxiety', 'anger,surprise', 'happiness,contempt' or 'anxiety,helplessness'."
        ).format(text_whisper=text_whisper)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# "MC-EIU_emotion": lambda: load_mc_eiu_emotion(model),
def load_mc_eiu_emotion(model=['model']):
    input_file = "/home/user/big_space/datasets/MC-EIU/MC-EIU_label.csv"
    video_dir = "/home/user/big_space/datasets/MC-EIU/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'validation'])]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mkv")
        label = row['emotion']
        text = row['Subtitle']
        mode = row['mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, fear, happy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text=text)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, fear, happy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text=text)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# "MC-EIU_intent":  lambda: load_mc_eiu_intent(model),
def load_mc_eiu_intent(model=['model']):
    input_file = "/home/user/big_space/datasets/MC-EIU/MC-EIU_label.csv"
    video_dir = "/home/user/big_space/datasets/MC-EIU/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'validation'])]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mkv")
        label = row['intent']
        text = row['Subtitle']
        mode = row['mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in **multimodal intent recognition**. "
            "Given the input video and audio, your task is to identify the **speaker's communicative intent** based on both visual and auditory information. "
            "Your output must be **only one intent label**, strictly chosen from the following list:\n"
            "acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text=text)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in **multimodal intent recognition**. "
            "Given the input video, your task is to identify the **speaker's communicative intent** based on both visual information. "
            "Your output must be **only one intent label**, strictly chosen from the following list:\n"
            "acknowledging, agreeing, consoling, encouraging, neutral, questioning, suggesting, wishing.\n"
            "Please ensure that the predicted label is **only from this list**.\n"
            "The person in the video says: {text} Based on the multimodal input, please identify the most likely intent conveyed by the speaker."
        ).format(text=text)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# "MELD_emotion":         lambda: load_meld_emotion(model),
def load_meld_emotion(model=['model']):
    input_file = "/home/user/big_space/datasets/MELD/MELD_label.csv"
    video_dir = "/home/user/big_space/datasets/MELD/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'dev'])]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['Emotion']
        text = row['text']
        mode = row['mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, fear, joy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text=text)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "anger, disgust, fear, joy, neutral, sad, surprise.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text=text)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# "MELD_sentiment":       lambda: load_meld_sentiment(model),
def load_meld_sentiment(model=['model']):
    input_file = "/home/user/big_space/datasets/MELD/MELD_label.csv"
    video_dir = "/home/user/big_space/datasets/MELD/video"
    df = pd.read_csv(input_file)
    df = df[df['mode'].isin(['test', 'dev'])]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['Sentiment']
        text = row['text']
        mode = row['mode']

        qwen_omni_prompt = f"""The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis."
            "Given the input video and audio, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."
        )
        
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion sentiment analysis."
            "Given the input video, your task is to identify the sentiment expressed by the person or people in the video. "
            "Your output must be only one sentiment label strictly chosen from the following list:\n"
            "Neutral, Negative, Positive.\n"
            "Please ensure that all predicted labels come only from this list."            
            "The person in the video says: {text}. Based on the multimodal input, please identify only one sentiment category expressed by the person in the video."
        ).format(text=text)
        qwen25_vl_system = ""

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result


# "MER2023":      lambda: load_mer2023(model),
def load_mer2023(model=['model']):
    input_file = "/home/user/big_space/datasets/MER2024/MER2024_label.csv"
    video_dir = "/home/user/big_space/datasets/MER2024/video"
    df = pd.read_csv(input_file)
    df = df[df['MER2023_mode'].isin(['test1', 'test2', 'test3'])]
    print(f"✅ 筛选后需要评测的样本数量: {len(df)} 行")

    result = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        label = row['emotion_label']
        text_whisper = row['chinese']
        mode = row['MER2023_mode']

        # 定义多个模型的 prompt 和 system
        qwen_omni_prompt = f"""The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."""
        qwen_omni_system = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video and audio, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "angry, happy, neutral, sad, surprise, worried.\n"
            "Please ensure that all predicted labels come only from this list."
        ).format(text_whisper=text_whisper)

        qwen25_vl_system = f""""""
        qwen25_vl_prompt = (
            "You are a helpful assistant specialized in multimodal emotion recognition. "
            "Given the input video, your task is to identify the emotion expressed by the person or people in the video. "
            "Your output must be only one emotion label strictly chosen from the following list:\n"
            "angry, happy, neutral, sad, surprise, worried.\n"
            "Please ensure that all predicted labels come only from this list."
            "The person in the video says: {text_whisper}. Based on the multimodal input, please identify only one emotion category expressed by the person in the video."
        ).format(text_whisper=text_whisper)

        # 根据 model 添加对应字段
        item = {
            "video_id": video_id,
            "mode": mode,
            "label": label,
            "video_path": video_path
        }

        if 'qwen_omni' in model:
            item["prompt"] = qwen_omni_prompt
            item["system"] = qwen_omni_system


        if 'qwen25_vl' in model:
            item["prompt"] = qwen25_vl_prompt
            item["system"] = qwen25_vl_system
   
        result.append(item)
    return result










