from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask
from minigpt4.models.emotion_llama_v2 import EmotionLLaMAv2


@registry.register_task("video_audio_pretrain")
class VideoAudioPretrainTask(BaseTask):
    def __init__(self):
        print("@registry.register_task: video_audio_pretrain")
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

    def build_model(self, cfg):
        return super().build_model(cfg)