import os
import sys
import random
import argparse
import numpy as np
import gradio as gr
from PIL import Image

import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, CONV_VISION_minigptv2, Chat

from minigpt4.datasets.builders import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.models import *
from minigpt4.tasks import *

VERSION = "1.0.0"

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/demo.yaml',
                        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

print("--- Initializing ---")
setup_seeds()


args = parse_args()
cfg = Config(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()

key = list(cfg.datasets_cfg.keys())[0]
vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


print('Initializing Chat Controller...')
chat = Chat(model, vis_processor, device=device)
print("--- Initializing finish ---")

def escape_markdown(text):
    md_chars = ['<', '>']
    for char in md_chars:
        text = text.replace(char, '\\' + char)
    return text

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list.clear()
    
    return (
        None,
        gr.update(value=None, interactive=True), 
        gr.update(placeholder='Upload your video and chat', interactive=True),
        None, 
        []    
    )


def gradio_user_ask(user_message, chatbot):
    if len(user_message) == 0:
        gr.Warning("Your question cannot be empty.")
        return chatbot, ""
    
    chatbot = chatbot + [[user_message, None]]
    return chatbot, ""

def gradio_bot_answer(chatbot, chat_state, gr_video, temperature):
    user_message = chatbot[-1][0]

    print("\n" + "="*50, flush=True)
    print("--- Bot Answer Function Entered ---", flush=True)
    print(f"  UI Video Path (gr_video): {gr_video}", flush=True)
    if chat_state and hasattr(chat_state, 'video_path'):
        print(f"  Stored Video Path (chat_state.video_path): {chat_state.video_path}", flush=True)
    else:
        print("  Stored Video Path (chat_state.video_path): None (This is the first run)", flush=True)

    if chat_state is None:
        chat_state = CONV_VISION_minigptv2.copy()
        
    is_new_video_session = not hasattr(chat_state, 'messages') or not chat_state.messages or chat_state.video_path != gr_video
    
    print(f"  Decision: Is this a new video session? -> {is_new_video_session}", flush=True)

    if is_new_video_session:
        print("  Action: Will reset chat and process NEW video features.", flush=True)
        chat_state = CONV_VISION_minigptv2.copy()
        chat_state.video_path = gr_video
    else:
        print("  Action: Will use CACHED video features from the previous turn.", flush=True)

    img_list = []
    try:
        if chat_state.needs_processing:
            print("  - Processing: Feature extraction started.", flush=True)
            temp_img_list = [chat_state.video_path]
            chat.encode_img(temp_img_list)
            chat_state.image_embeds = temp_img_list[0]
            img_list.append(chat_state.image_embeds)
            chat_state.needs_processing = False
            print("  - Processing: Feature extraction complete.", flush=True)
        else:
            img_list.append(chat_state.image_embeds)
    except Exception as e:
        print(f"  - ERROR: {e}", flush=True)
        gr.Error(f"An error occurred while processing the video: {e}")
        chatbot[-1][1] = f"Error: {e}"
        yield chatbot, chat_state
        return

    chat.ask(user_message, chat_state, is_first_turn=is_new_video_session)
    chatbot[-1][1] = ""

    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=temperature,
                                  max_new_tokens=500,
                                  max_length=2000)


    output = ""
    is_new_sentence = True
    sentence_enders = (".", "?", "!")

    for new_output in streamer:
        escaped_new_output = escape_markdown(new_output)
        
        if is_new_sentence and len(escaped_new_output.strip()) > 0:
            stripped_chunk = escaped_new_output.lstrip()
            capitalized_chunk = stripped_chunk[0].upper() + stripped_chunk[1:]
            
            leading_whitespace = escaped_new_output[:len(escaped_new_output) - len(stripped_chunk)]
            escaped_new_output = leading_whitespace + capitalized_chunk
            
            is_new_sentence = False

        output += escaped_new_output

        if output.rstrip().endswith(sentence_enders):
            is_new_sentence = True

        chatbot[-1][1] = output
        yield chatbot, chat_state

    if output and not output.rstrip().endswith(sentence_enders):
        output += '.'
    
    chatbot[-1][1] = output
    chat_state.messages[-1][1] = output
    
    print("--- Bot Answer Function Finished ---", flush=True)
    print("="*50 + "\n", flush=True)

    yield chatbot, chat_state

    return chatbot, chat_state


title = """<h1 align="center">Emotion-LLaMA-v2 Demo</h1>"""
introduction = '''
**How to use:**
1. Upload a video.
2. Type your question in the textbox.
3. Click "Send" or press Enter.
4. To analyze a new video, simply upload another one. The chat will automatically restart.
'''

quick_prompts = {
    "Emotion": "[emotion] Identify the emotion expressed in the video. Choose from: anger, disgust, fear, happy, neutral, sad, surprise.",
    "Sentiment": "[sentiment] Identify the displayed sentiment in the video: Positive, Neutral or Negative.",
    "Reason": "[reason] Why does the person in the video feel this way? Please provide evidence from the video.",
    "Think": "[think] The possible emotions are: Anger, Happiness, Disgust, Surprise, Fear, Sadness, Excited, Neutral, Frustration. Analyze the person's emotional state in the video by combining visual, auditory, and textual clues. Provide your reasoning in <think> and your final label in <answer>.",
}

with gr.Blocks(theme=gr.themes.Soft(), css="#chatbot .user {background-color: #EBF5FB;} #chatbot .bot {background-color: #E8F8F5;}") as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            temperature_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.2, step=0.1, interactive=True, label="Temperature")

            with gr.Accordion("Quick Prompts", open=True):
                gr.Markdown("Click a button to use a predefined prompt.")
                with gr.Row():
                    emotion_btn = gr.Button("Emotion")
                    sentiment_btn = gr.Button("Sentiment")
                with gr.Row():
                    reason_btn = gr.Button("Reason")
                    think_btn = gr.Button("Think")

            clear_button = gr.Button("🔄 Restart Conversation")
            
        with gr.Column(scale=2):
            chatbot_display = gr.Chatbot(label='Emotion-LLaMA-v2', height=600, elem_id="chatbot")
            
            with gr.Row():
                text_input_box = gr.Textbox(placeholder='Type your question here...', show_label=False, container=False, scale=8)
                send_button = gr.Button("▶️ Send", variant='primary', size='sm', scale=1)

    gr.Markdown(introduction)
    
    examples = [
        ["examples/Ses05F_impro03_F000.mp4", "[emotion] Identify the emotion expressed in the video. Choose from: anger, disgust, fear, happy, neutral, sad, surprise."],
        ["examples/sample_00000033.mp4", "[sentiment] Identify the displayed sentiment in the video: Positive, Neutral or Negative."],
        ['examples/dev_dia1_utt3.mp4',"The person in video says: Now, theres two reasons. [reason] Taking into account the speaker's expressions, vocal tone, and spoken words, what sentiment—Neutral, Positive, or Negative—can be inferred from the combined cues?"],
        ["examples/77.mp4","The person in video says: Look who's here. [think] Based on the speaker’s visual and auditory behavior in the video, choose one emotion from: happy, anger, neutral, surprise, fear, sad, disgust. In <think>, describe the visual cues (e.g., eye contact, gestures), the vocal tone (e.g., stressed, calm), and the meaning of what is said. Then, in <answer>, provide your final label."]

    ]
    gr.Examples(examples=examples, inputs=[video_input, text_input_box])

    chat_state = gr.State()
    img_list = gr.State([]) 


    user_ask_inputs = [text_input_box, chatbot_display]
    user_ask_outputs = [chatbot_display, text_input_box]
    
    bot_answer_inputs = [chatbot_display, chat_state, video_input, temperature_slider]
    bot_answer_outputs = [chatbot_display, chat_state]
    
    submit_action = send_button.click if not gr.Number else text_input_box.submit

    send_button.click(
        fn=gradio_user_ask,
        inputs=user_ask_inputs,
        outputs=user_ask_outputs,
        queue=False
    ).then(
        fn=gradio_bot_answer,
        inputs=bot_answer_inputs,
        outputs=bot_answer_outputs
    )

    text_input_box.submit(
        fn=gradio_user_ask,
        inputs=user_ask_inputs,
        outputs=user_ask_outputs,
        queue=False
    ).then(
        fn=gradio_bot_answer,
        inputs=bot_answer_inputs,
        outputs=bot_answer_outputs
    )

    clear_button.click(
        fn=gradio_reset,
        inputs=[chat_state, img_list],
        outputs=[chatbot_display, video_input, text_input_box, chat_state, img_list],
        queue=False
    )

    emotion_btn.click(lambda: quick_prompts["Emotion"], None, text_input_box)
    sentiment_btn.click(lambda: quick_prompts["Sentiment"], None, text_input_box)
    reason_btn.click(lambda: quick_prompts["Reason"], None, text_input_box)
    think_btn.click(lambda: quick_prompts["Think"], None, text_input_box)

demo.queue().launch(share=True,server_name="0.0.0.0")
