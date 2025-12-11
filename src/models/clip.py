import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base
from transformers import AutoModel
import numpy as np

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        try: backbone = convnext_base(weights="IMAGENET1K_V1")
        except: backbone = convnext_base(pretrained=True)
            
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(15, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 15, 1, 1)
            
        backbone.features[0][0] = new_conv
        self.backbone = backbone.features 
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(1024, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x).flatten(1)
        return F.normalize(self.global_proj(x), dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, model_name, embed_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.global_proj = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        return F.normalize(self.global_proj(cls_emb), dim=-1)

class SpineCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        clip_cfg = cfg['models']['clip']
        self.img_enc = ImageEncoder(embed_dim=clip_cfg['embed_dim'])
        self.txt_enc = TextEncoder(clip_cfg['text_encoder'], embed_dim=clip_cfg['embed_dim'])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.img_enc(images)
        txt_emb = self.txt_enc(input_ids, attention_mask)
        
        # [핵심 수정] Forward 안에서 scale을 적용하여 리턴
        # 이렇게 해야 DDP가 logit_scale 사용을 인지함
        return img_emb, txt_emb, self.logit_scale.exp()