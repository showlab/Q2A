import torch
from torch import nn
from pytorch_lightning import LightningModule
import timm, os
from transformers import AutoModel

class Encoder(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.vision_model = timm.create_model(cfg.MODEL.VISION, pretrained=True)
        self.vision_model.head = nn.Identity()

        self.text_model = AutoModel.from_pretrained(cfg.MODEL.TEXT)
        self.for_video = cfg.FOR.VIDEO
        self.for_script = cfg.FOR.SCRIPT
        self.for_qa = cfg.FOR.QA

    def test_step(self, batch, idx):
        if batch[0] is None:
            return 
        
        if self.for_video:
            video, path = batch[0] 
            # set smaller batch size to prevent OOM
            # features = torch.cat([
            #     self.vision_model(frame.unsqueeze(0))
            #     for frame in video
            # ])
            features = self.vision_model(video)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(features, os.path.join(path, "video.pth"))
        if self.for_script:
            script, path = batch[0]
            if not os.path.exists(path):
                os.makedirs(path)
            features = torch.cat([self.text_model(**sentence).pooler_output for sentence in script])
            torch.save(features, os.path.join(path, "script.pth"))
        if self.for_qa:
            qas, path, tag = batch[0]
            if not os.path.exists(path):
                os.makedirs(path)
            for qa in qas:
                qa['question'] = self.text_model(**qa['question']).pooler_output
                button_features = []
                for button_images_per_step in qa['button_images']:
                    button_features.append(
                        [
                            self.vision_model(button_image.view(-1,3,button_image.shape[-2], button_image.shape[-1])).flatten() \
                            for button_image in button_images_per_step
                        ]
                    )
                for i, answers_per_step in enumerate(qa['answers']):
                    for j, answer in enumerate(answers_per_step):
                        bidx = qa['answer_bidxs'][i][j]
                        button_feature = button_features[i][bidx]
                        text_feature = self.text_model(**answer).pooler_output
                        answer_feature = dict(text=text_feature, button=button_feature)
                        qa['answers'][i][j] = answer_feature
            torch.save(qas, os.path.join(path, f'{tag}.pth'))

def build_model(cfg):
    return Encoder(cfg)
