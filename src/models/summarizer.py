# src/models/summarizer.py
import torch.nn as nn
from transformers import BartForConditionalGeneration, AutoConfig

class ReportSummarizer(nn.Module):
    def __init__(self, model_name="GanjinZero/biobart-base"):
        super().__init__()
        # BioBART는 BART 구조임
        try:
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        except:
            # 혹시라도 로드 실패시 Config로 초기화 시도
            config = AutoConfig.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)