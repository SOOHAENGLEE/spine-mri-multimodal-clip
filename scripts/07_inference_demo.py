# scripts/07_inference_demo.py
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import cv2
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.data.datasets import CLIPDataset
from src.models.clip import SpineCLIP

class GradCAM:
    """ConvNeXt용 간단한 Grad-CAM 구현"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, img_tensor, target_embedding):
        # Forward
        self.model.zero_grad()
        img_emb = self.model(img_tensor) # (1, 512)
        
        # Target Embedding과의 내적을 Score로 설정 (유사도가 높을수록 활성화)
        score = (img_emb * target_embedding).sum()
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0] # (C, H, W)
        activations = self.activations.cpu().data.numpy()[0] # (C, H, W)
        
        # Global Average Pooling of Gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) # Normalize 0~1
        return cam

def visualize_inference(model, dataset, device, num_samples=3, k=3, save_dir="./results/demo"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 갤러리 텍스트 임베딩 미리 계산
    print("Pre-computing text embeddings...")
    all_texts = []
    all_txt_embs = []
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            # SpineCLIP 구조에 따라 txt_enc 호출
            emb = model.txt_enc(ids, mask) 
            all_txt_embs.append(emb.cpu())
            all_texts.extend(batch['raw_text'])
    
    gallery_embs = torch.cat(all_txt_embs, dim=0).to(device)
    
    # Grad-CAM 준비 (ImageEncoder의 마지막 feature block)
    # model.img_enc.backbone은 nn.Sequential 형태임
    target_layer = list(model.img_enc.backbone.modules())[-1]
    grad_cam = GradCAM(model.img_enc, target_layer)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f"Visualizing {num_samples} samples...")
    for i, idx in enumerate(indices):
        data = dataset[idx]
        img_tensor = data['images'].unsqueeze(0).to(device)
        gt_text = data['raw_text']
        
        # 1. Retrieval
        with torch.no_grad():
            img_emb = model.img_enc(img_tensor)
            sims = img_emb @ gallery_embs.t()
            scores, top_idxs = sims.squeeze(0).topk(k)
        
        # 2. Grad-CAM (Top-1 텍스트에 대해 이미지가 어디를 보는지)
        # Gradient 계산을 위해 enable_grad
        top1_txt_emb = gallery_embs[top_idxs[0]].unsqueeze(0)
        heatmap = grad_cam(img_tensor, top1_txt_emb)
        
        # 3. Plotting
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])
        
        # (1) 원본 이미지 (중간 슬라이스)
        ax1 = fig.add_subplot(gs[0])
        mid_slice = data['images'][7].cpu().numpy() * 0.5 + 0.5
        ax1.imshow(mid_slice, cmap='gray')
        ax1.set_title(f"Query MRI (Slice 7)\nIdx: {idx}")
        ax1.axis('off')
        
        # (2) Grad-CAM Heatmap
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(mid_slice, cmap='gray')
        ax2.imshow(heatmap, cmap='jet', alpha=0.5) # Overlay
        ax2.set_title("Grad-CAM Activation\n(Where model looks)")
        ax2.axis('off')
        
        # (3) 검색 결과 텍스트 (Wrapping 적용)
        ax3 = fig.add_subplot(gs[2])
        result_txt = f"🔍 [Top-{k} Retrieval Results]\n" + "-"*40 + "\n"
        
        for rank, txt_idx in enumerate(top_idxs):
            score = scores[rank].item()
            retrieved = all_texts[txt_idx]
            mark = "✅ CORRECT" if retrieved == gt_text else ""
            
            # Text Wrap (중요)
            wrapped_lines = textwrap.wrap(retrieved, width=60) 
            formatted_txt = "\n   ".join(wrapped_lines)
            
            result_txt += f"{rank+1}. [{score:.3f}] {mark}\n   {formatted_txt}\n\n"
            
        # Ground Truth 표시
        gt_wrapped = textwrap.fill(gt_text, width=60)
        result_txt += f"\n📝 [Ground Truth]\n{gt_wrapped}"
        
        ax3.text(0.02, 0.98, result_txt, fontsize=11, va='top', fontfamily='monospace')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/demo_sample_{i}.png")
        plt.close()
        print(f"Saved {save_dir}/demo_sample_{i}.png")

def main():
    TASK_MODE = "summary"
    # 경로 설정
    MODEL_PATH = f"./results/clip_{TASK_MODE}/best_model.pt"
    DATA_PATH = "./data/processed/augmented.json"
    BERT_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    ds = CLIPDataset(DATA_PATH, tokenizer, text_col=TASK_MODE, is_train=False)
    
    # 모델 로드
    model = SpineCLIP({'models': {'clip': {'embed_dim': 512, 'text_encoder': BERT_NAME}}})
    model = model.to(device)
    
    if os.path.exists(MODEL_PATH):
        # DDP로 저장된 경우 'module.' 제거 처리
        state_dict = torch.load(MODEL_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print(f"✅ Loaded model from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Exiting...")
        return

    visualize_inference(model, ds, device, num_samples=5, k=3)

if __name__ == "__main__":
    main()