import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.models.convpro_model import ConvProAttention
import re


@registry.register_model("emotion_llama_v2")
class EmotionLLaMAv2(MiniGPTBase):
    """
    EmotionLLaMA v2 model, extending MiniGPTBase with new emotion reasoning abilities.
    """
    print("Emotion-LLaMA-v2 model, extending MiniGPTBase with new emotion reasoning abilities.")

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/emotion_llama_v2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=3800,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        img_f_dim = self.visual_encoder.num_features * 4

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        
        self.llama_proj_video = nn.Linear(
            1408, self.llama_model.config.hidden_size
        )
        self.llama_proj_audio = nn.Linear(
            1280, self.llama_model.config.hidden_size
        )    

        conv_input_dims = [1280, 1408, 1408] 
        conv_internal_dim = 128 

        conv_output_dim = 256 

        self.conv_pro_module = ConvProAttention(
            dims=conv_input_dims,
            output_dim=conv_output_dim,
            layers=f'512,{conv_internal_dim}', # MLP 结构: 1xxx -> 512 -> 128
            dropout=0.1
        )

        self.llama_proj_fusion = nn.Linear(
            conv_output_dim, self.llama_model.config.hidden_size
        )

        self.debug = False

        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

    def encode_img(self, image, video_features, audio_features):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            raw_image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)        # [B, 1025, 1408]

            image_embeds = raw_image_embeds[:, 1:, :]                                       # [B, 1024, 1408]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))              # [B, 256, 5632]

            img_cls = raw_image_embeds[:, 0, :]
            img_seq_64 = img_cls.unsqueeze(1).expand(-1, 64, -1)

            aud_seq_64 = audio_features.view(bs, 64, -1)
            vid_seq_64 = video_features.view(bs, 64, -1)

            fusion_out, _ = self.conv_pro_module([aud_seq_64, img_seq_64, vid_seq_64]) 

            fusion_inputs_llama = self.llama_proj_fusion(fusion_out)

            image_inputs_llama = self.llama_proj(image_embeds)  # [B, 256, llama_hidden]

            video_inputs_llama = self.llama_proj_video(
                video_features.view(bs, 64, 1408)
            )  # [B, 64, llama_hidden]

            audio_inputs_llama = self.llama_proj_audio(
                audio_features.view(bs, 64, 1280)
            )  # [B, 64, llama_hidden]
            
            # concat along sequence dim: [B, 256+64+64, llama_hidden]
            inputs_llama = torch.cat(
                (fusion_inputs_llama, image_inputs_llama, video_inputs_llama, audio_inputs_llama), dim=1
            )
                
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

            image_all_zero = (image.abs().sum(dim=list(range(1, image.dim()))) == 0)  
            video_all_zero = (video_features.abs().sum(dim=list(range(1, video_features.dim()))) == 0)  
            audio_all_zero = (audio_features.abs().sum(dim=list(range(1, audio_features.dim()))) == 0)  

            atts_llama[image_all_zero, 64:320] = 0
            atts_llama[video_all_zero, 320:384] = 0
            atts_llama[audio_all_zero, 384:] = 0

            if self.debug == False:
                print('=========================================')
                print(f"Fusion Input Shapes - Audio: {aud_seq_64.shape}, Img: {img_seq_64.shape}, Vid: {vid_seq_64.shape}")
                print(f"Fusion Output Shape: {fusion_inputs_llama.shape}")
                print("Total inputs_llama.shape:", inputs_llama.shape)
                print('=========================================')
                self.debug = True
                
        return inputs_llama, atts_llama


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 3800)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Emotion-LLaMA Checkpoint: {}".format(ckpt_path))

            state_dict = torch.load(ckpt_path)['model']
            filtered = {k: v for k, v in state_dict.items() if k in model.state_dict()}
            model.load_state_dict(filtered, strict=False)

        return model
