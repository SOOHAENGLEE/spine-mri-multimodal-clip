# scripts/09_viz_generator_demo.py
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from accelerate import Accelerator
import argparse # 🔥 [수정] argparse import 추가

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.data.datasets import CLIPDataset
from src.models.generator import MRIReportGenerator

def visualize_generation_samples(
    model, 
    tokenizer, 
    dataset, 
    subset_indices, 
    device, 
    num_samples=5, 
    save_path="./results/viz_generator_demo.png"
):
    model.eval()
    n_total = len(subset_indices)
    query_indices = np.random.choice(n_total, num_samples, replace=False)
    
    fig = plt.figure(figsize=(16, 6 * num_samples))
    gs = fig.add_gridspec(num_samples, 2, width_ratios=[1, 2.5])

    print(f"   -> Generating summaries for {num_samples} samples...")
    
    for i, q_idx in enumerate(query_indices):
        real_query_idx = subset_indices[q_idx]
        query_data = dataset[real_query_idx]
        img_tensor = query_data['images'][7]
        gt_summary = query_data['raw_text']
        
        img_batch = query_data['images'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            unwrapped_model = model
            gen_ids = unwrapped_model.generate(
                img_batch, 
                max_length=128, 
                num_beams=4
            )
        gen_summary = tokenizer.decode(gen_ids[0].cpu(), skip_special_tokens=True)
        
        ax_img = fig.add_subplot(gs[i, 0])
        disp_img = img_tensor.cpu().numpy() * 0.5 + 0.5
        ax_img.imshow(disp_img, cmap='gray')
        ax_img.set_title(f"Query MRI (Idx: {real_query_idx})", fontsize=14, fontweight='bold')
        ax_img.axis('off')

        ax_txt = fig.add_subplot(gs[i, 1])
        wrapper = textwrap.TextWrapper(width=80)
        wrapped_generated = "\n".join(wrapper.wrap(gen_summary))
        wrapped_gt = "\n".join(wrapper.wrap(gt_summary))
        
        text_content = (
            f"🤖 [Generated Summary]:\n{wrapped_generated}\n\n"
            f"{'-'*70}\n"
            f"📝 [Ground Truth Summary]:\n{wrapped_gt}"
        )
        ax_txt.text(0.02, 0.95, text_content, fontsize=11, va='top', fontfamily='monospace', linespacing=1.6)
        ax_txt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Saved Generation Demo: {save_path}")

def main(args): # 🔥 [수정] args 인자 추가
    accelerator = Accelerator()
    device = accelerator.device

    with open(args.config, 'r') as f: 
        cfg = yaml.safe_load(f)

    GEN_MODEL_PATH = f"{cfg['paths']['result_dir']}/image_to_summary/best_model.pt"
    DATA_PATH = f"{cfg['paths']['output_dir']}/augmented.json"
    MODEL_NAME = cfg['models']['summarizer']['name']
    
    if accelerator.is_main_process:
        print("🚀 Running Image-to-Summary Generation Visualization...")
        os.makedirs("./results", exist_ok=True)
        
        if not os.path.exists(GEN_MODEL_PATH):
            print(f"❌ Generator model checkpoint not found at {GEN_MODEL_PATH}. Run Step 6 first.")
            return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    fallback_path = f"{cfg['paths']['output_dir']}/labeled.json"
    if not os.path.exists(DATA_PATH):
        if os.path.exists(fallback_path):
            DATA_PATH = fallback_path
            if accelerator.is_main_process:
                print(f"⚠️ Warning: augmented.json not found. Falling back to labeled.json.")
        else:
             raise FileNotFoundError(f"Required data file not found at {DATA_PATH} or {fallback_path}")

    ds = CLIPDataset(DATA_PATH, tokenizer, text_col='summary', is_train=False)
    
    n_samples = min(200, len(ds)) 
    indices = np.random.choice(len(ds), n_samples, replace=False)
    subset_indices = indices.tolist()
    
    model = MRIReportGenerator(text_model_name=MODEL_NAME).to(device)
    state = torch.load(GEN_MODEL_PATH, map_location=device)
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    print("✅ Generator Model Loaded")
    
    if accelerator.is_main_process:
        visualize_generation_samples(model, tokenizer, ds, subset_indices, device, num_samples=5, save_path="./results/viz_generator_demo.png")
        print("✨ Generation Visualization done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args) # 🔥 [수정] args 전달