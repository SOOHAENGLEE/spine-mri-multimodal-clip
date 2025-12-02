import cv2
import torch
import json
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as T

# [Deadlock Prevention] OpenCV 멀티스레딩 끄기
cv2.setNumThreads(0)

class SummarizationDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_input_len=512, max_target_len=128):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_input = max_input_len
        self.max_target = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item.get('full_report', "") or "", max_length=self.max_input, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(item.get('summary', "") or "", max_length=self.max_target, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

class CLIPDataset(Dataset):
    def __init__(self, json_path, tokenizer, img_size=(320, 320), text_col="summary", max_len=128, is_train=True):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_len = max_len
        self.is_train = is_train
        
        # Tensor 상태에서의 Augmentation (이미 Normalize된 상태라 조심스럽게 적용)
        if self.is_train:
            self.transform = T.Compose([
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = nn.Identity()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # [Speedup] .pt 로드 (매우 빠름)
        try:
            images = torch.load(item['image_path']) # (15, H, W)
            if self.is_train:
                images = self.transform(images)
        except Exception as e:
            # 학습 중단 방지를 위한 Fallback
            print(f"🚨 Load Error: {item['image_path']} | {e}")
            images = torch.zeros(15, 320, 320)

        txt = item.get(self.text_col, "") or ""
        full_report = item.get('full_report', "") or ""
        enc = self.tokenizer(str(txt), max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "images": images,
            "input_ids": enc.input_ids.squeeze(),
            "attention_mask": enc.attention_mask.squeeze(),
            "raw_text": str(txt),
            "full_report": str(full_report)
        }