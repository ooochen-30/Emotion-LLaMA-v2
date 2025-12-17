import os
import torch
import argparse
from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from my_utils.extract_features import FeatureExtractor, extract_frame

import warnings

warnings.filterwarnings("ignore")

def run_single_inference(video_path, instruction, model, task, vis_processor, feature_extractor):
    
    image = extract_frame(video_path)
    if image is None:
        print(f"ERROR: Could not extract frame from {video_path}. Skipping.")
        return None
    image_tensor = vis_processor(image).unsqueeze(0).to(feature_extractor.device)
    
    # The encoder needs to be loaded during the first round of inference
    video_feature = feature_extractor.extract_eva_vit_g_features(video_path)
    audio_feature = feature_extractor.extract_whisper_audio_features(video_path)

    video_feature = video_feature.to(feature_extractor.device)
    audio_feature = audio_feature.to(feature_extractor.device)

    conv = CONV_VISION_minigptv2.copy()
    conv.system = ""

    full_instruction = task + " " + instruction
    texts = prepare_texts([full_instruction], conv)
    
    with torch.no_grad():
        response = model.generate(
            image_tensor, 
            video_feature, 
            audio_feature, 
            texts, 
            max_new_tokens=200, 
            do_sample=False
        )[0]
    
    return response


if __name__ == "__main__":
    parser = eval_parser()
    args = parser.parse_args()
    cfg = Config(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, vis_processor = init_model(args)
    model.eval()
    
    feature_extractor = FeatureExtractor(device=device)

    while True:
        try:
            video_path = input("Enter the path to your video file (or type 'exit' to quit): ")
            if video_path.lower() == 'exit':
                break
            if not os.path.exists(video_path):
                print("  File not found. Please try again.\n")
                continue

            instruction = input("Enter your question: ")
            
            task = "[emotion]"
            
            response = run_single_inference(video_path, instruction, model, task, vis_processor, feature_extractor)

            if response:
                print("\n[Emotion-LLaMA-v2]:")
                print(response)
                print("-" * 20 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            continue 