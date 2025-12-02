# src/models/generator.py
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from torchvision.models import convnext_base

class ImageEncoder25D(nn.Module):
    def __init__(self, embed_dim=768, in_channels=15):
        super().__init__()
        # ConvNeXt Base 로드
        try: backbone = convnext_base(weights="IMAGENET1K_V1")
        except: backbone = convnext_base(pretrained=True)
        
        # 첫 레이어 채널 수정 (3 -> 15)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, 
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                             padding=old_conv.padding)
        # 가중치 복사 (평균)
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            
        backbone.features[0][0] = new_conv
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1024, embed_dim) # ConvNeXt Base output=1024

    def forward(self, x):
        x = self.backbone(x) # (B, 1024, H, W)
        x = self.pool(x).flatten(1) # (B, 1024)
        return self.proj(x) # (B, embed_dim)

class MRIReportGenerator(nn.Module):
    def __init__(self, text_model_name="gogamza/kobart-base-v2"):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(text_model_name)
        self.img_encoder = ImageEncoder25D(embed_dim=self.bart.config.d_model, in_channels=15)

    def forward(self, images, labels=None):
        # images: (B, 15, H, W)
        img_embeds = self.img_encoder(images).unsqueeze(1) # (B, 1, Hidden)
        
        if labels is not None:
            return self.bart(inputs_embeds=None, encoder_outputs=(img_embeds,), labels=labels)
        else:
            return self.bart.generate(encoder_outputs=(img_embeds,), max_length=128)