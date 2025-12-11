import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
from accelerate import Accelerator

# UMAP 라이브러리 확인 (없으면 t-SNE 사용)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.data.datasets import CLIPDataset
from src.models.clip import SpineCLIP
from src.models.summarizer import ReportSummarizer

def visualize_similarity_matrix(sim_matrix, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=False, cmap="viridis")
    plt.title("Image-Text Similarity Matrix (32x32 Batch)")
    plt.xlabel("Text Index")
    plt.ylabel("Image Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Saved Heatmap: {save_path}")

def visualize_retrieval_best_match(
    img_embs, 
    txt_embs, 
    dataset, 
    subset_indices, 
    summ_model, 
    summ_tokenizer, 
    device, 
    num_samples=5, 
    save_path="./results/viz_retrieval.png"
):
    """
    Query Image에 대해 가장 유사도가 높은(Hit@1) Report를 찾고,
    그 Report를 Summarizer로 요약하여 시각화합니다.
    """
    summ_model.eval()
    
    # 1. 랜덤한 Query 샘플 선택
    n_total = len(img_embs)
    query_indices = np.random.choice(n_total, num_samples, replace=False)
    
    fig = plt.figure(figsize=(16, 6 * num_samples))
    gs = fig.add_gridspec(num_samples, 2, width_ratios=[1, 2]) # 이미지 1 : 텍스트 2 비율

    print(f"   -> Generating Hit@1 summaries for {num_samples} samples...")
    
    for i, q_idx in enumerate(query_indices):
        # --- 1. Retrieval (검색) ---
        query_img_emb = torch.tensor(img_embs[q_idx]).to(device) # (D,)
        
        gallery_txt_embs = torch.tensor(txt_embs).to(device)
        sims = torch.matmul(gallery_txt_embs, query_img_emb) # (N,)
        
        best_match_idx = torch.argmax(sims).item()
        
        real_query_idx = subset_indices[q_idx]
        real_retrieved_idx = subset_indices[best_match_idx]
        
        # --- 2. Data Fetching ---
        query_data = dataset[real_query_idx]
        img_tensor = query_data['images'][7] # 중간 슬라이스 (Visual용)
        
        retrieved_data = dataset[real_retrieved_idx]
        retrieved_full_report = retrieved_data['full_report']
        retrieved_score = sims[best_match_idx].item()
        
        is_correct = (q_idx == best_match_idx)
        match_status = "✅ Correct Match" if is_correct else f"⚠️ Retrieved (Idx: {real_retrieved_idx})"

        # --- 3. Summarization (요약 생성) ---
        inputs = summ_tokenizer(
            retrieved_full_report, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        ).to(device)
        
        with torch.no_grad():
            gen_ids = summ_model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=128, 
                num_beams=4
            )
        gen_summary = summ_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        # --- 4. Plotting (시각화) ---
        ax_img = fig.add_subplot(gs[i, 0])
        disp_img = img_tensor.cpu().numpy() * 0.5 + 0.5 # Denormalize
        ax_img.imshow(disp_img, cmap='gray')
        ax_img.set_title(f"Query Image (Idx: {real_query_idx})", fontsize=14, fontweight='bold')
        ax_img.axis('off')

        ax_txt = fig.add_subplot(gs[i, 1])
        
        wrapper = textwrap.TextWrapper(width=70)
        wrapped_summary = "\n".join(wrapper.wrap(gen_summary))
        wrapped_original = "\n".join(wrapper.wrap(retrieved_full_report[:200] + "...")) # 너무 기니까 일부만
        
        text_content = (
            f"🔍 [Retrieval Result]: {match_status} (Score: {retrieved_score:.4f})\n"
            f"{'-'*60}\n"
            f"🤖 [Generated Summary of Hit@1 Report]:\n"
            f"{wrapped_summary}\n\n"
            f"{'-'*60}\n"
            f"📄 [Original Retrieved Report (Partial)]:\n"
            f"{wrapped_original}"
        )
        
        ax_txt.text(0.02, 0.95, text_content, fontsize=12, va='top', fontfamily='monospace', linespacing=1.6)
        ax_txt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Saved Retrieval Viz: {save_path}")

def visualize_projection(img_embs, txt_embs, save_path):
    """임베딩 분포 시각화 (UMAP/t-SNE)"""
    combined = np.concatenate([img_embs, txt_embs], axis=0)
    n_img = len(img_embs)
    
    plt.figure(figsize=(10, 10))
    
    if HAS_UMAP and n_img > 50:
        print("   -> Using UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(combined)
        title = "UMAP Projection"
    else:
        print("   -> Using t-SNE...")
        perp = min(30, n_img - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        proj = reducer.fit_transform(combined)
        title = "t-SNE Projection"

    plt.scatter(proj[:n_img, 0], proj[:n_img, 1], c='blue', alpha=0.6, label='Image', s=15)
    plt.scatter(proj[n_img:, 0], proj[n_img:, 1], c='red', alpha=0.6, label='Text', s=15)
    
    # 대응되는 쌍 선 연결 (일부만)
    if n_img <= 100:
        for k in range(n_img):
            plt.plot([proj[k, 0], proj[n_img+k, 0]], [proj[k, 1], proj[k, 1]], 'gray', alpha=0.1)

    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Saved Projection: {save_path}")

def main():
    accelerator = Accelerator()
    device = accelerator.device

    with open("configs/model_config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    TASK_MODE = "summary"
    CLIP_PATH = f"{cfg['paths']['result_dir']}/clip_{TASK_MODE}/best_model.pt"
    SUMM_PATH = f"{cfg['paths']['result_dir']}/summarizer/best_model.pt"
    
    # ====================================================================
    # 🔥 [수정] 데이터 경로 폴백 로직 추가 (augmented.json -> labeled.json)
    # ====================================================================
    DATA_PATH = f"{cfg['paths']['output_dir']}/augmented.json"
    fallback_path = f"{cfg['paths']['output_dir']}/labeled.json"
    
    if not os.path.exists(DATA_PATH):
        if os.path.exists(fallback_path):
            DATA_PATH = fallback_path
            if accelerator.is_main_process:
                print(f"⚠️ Warning: augmented.json not found. Falling back to labeled.json.")
        else:
             # labeled.json도 없으면 FileNotFoundError 발생
             raise FileNotFoundError(f"Required data file not found at {DATA_PATH} or {fallback_path}")
    
    BERT_NAME = cfg['models']['clip']['text_encoder']
    SUMM_NAME = cfg['models']['summarizer']['name']
    
    if accelerator.is_main_process:
        print("🚀 Running Final Visualization...")
        os.makedirs("./results", exist_ok=True)

    # 1. Dataset Load
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    ds = CLIPDataset(DATA_PATH, tokenizer, text_col=TASK_MODE, is_train=False)
    
    # 시각화용 Subset (500개 정도면 충분)
    n_samples = min(500, len(ds))
    indices = np.random.choice(len(ds), n_samples, replace=False)
    subset_indices = indices.tolist()
    subset = Subset(ds, subset_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. Model Load
    clip_model = SpineCLIP(cfg).to(device)
    if os.path.exists(CLIP_PATH):
        # DDP 호환 로드
        state = torch.load(CLIP_PATH, map_location=device)
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        clip_model.load_state_dict(new_state, strict=False)
        print("✅ CLIP Model Loaded")
    
    summ_model = ReportSummarizer(SUMM_NAME).to(device)
    summ_tokenizer = AutoTokenizer.from_pretrained(SUMM_NAME)
    if os.path.exists(SUMM_PATH):
        state = torch.load(SUMM_PATH, map_location=device)
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        summ_model.load_state_dict(new_state, strict=False)
        print("✅ Summarizer Model Loaded")

    clip_model.eval()
    summ_model.eval()

    # 3. Extract Embeddings
    img_embs_list, txt_embs_list = [], []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['images'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            i_emb, t_emb, _ = clip_model(imgs, ids, mask)
            
            img_embs_list.append(i_emb.cpu())
            txt_embs_list.append(t_emb.cpu())
            
    # Concat
    img_embs_np = torch.cat(img_embs_list, dim=0).numpy()
    txt_embs_np = torch.cat(txt_embs_list, dim=0).numpy()
    
    if accelerator.is_main_process:
        # 1. Similarity Matrix
        sim_matrix = img_embs_np @ txt_embs_np.T
        visualize_similarity_matrix(sim_matrix[:32, :32], "./results/viz_heatmap.png")
        
        # 2. Projection (UMAP/t-SNE)
        visualize_projection(img_embs_np, txt_embs_np, "./results/viz_projection.png")
        
        # 3. Retrieval & Generation Visualization
        visualize_retrieval_best_match(
            img_embs_np, 
            txt_embs_np, 
            ds, 
            subset_indices, 
            summ_model, 
            summ_tokenizer, 
            device, 
            num_samples=5, 
            save_path="./results/viz_retrieval.png"
        )
        print("✨ All done!")

if __name__ == "__main__":
    main()