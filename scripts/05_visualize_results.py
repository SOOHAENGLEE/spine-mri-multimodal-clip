# scripts/05_visualize_results.py
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml

def load_history(result_dir):
    json_path = os.path.join(result_dir, "history.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return pd.DataFrame(json.load(f))
    return None

def main():
    # Config 로드해서 결과 경로 추적
    config_path = "configs/model_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        BASE_DIR = cfg['paths']['result_dir']
    else:
        BASE_DIR = "./results"

    experiments = {
        "Summarizer (Task 1)": f"{BASE_DIR}/summarizer",
        "Proposed CLIP (Summary)": f"{BASE_DIR}/clip_summary",
        "Baseline CLIP (Full)": f"{BASE_DIR}/clip_full_report"
    }
    
    save_dir = f"{BASE_DIR}/comparison"
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # 1. CLIP Accuracy 비교 (Proposed vs Baseline)
    plt.figure(figsize=(10, 6))
    for name, path in experiments.items():
        if "Summarizer" in name: continue # CLIP만 비교
        
        df = load_history(path)
        if df is not None:
            # Key 호환성 (val_acc, acc1 등)
            acc_key = next((k for k in ['val_acc', 'acc1', 'hit1'] if k in df.columns), None)
            if acc_key:
                plt.plot(df['epoch'], df[acc_key], marker='o', label=f"{name}")
    
    plt.title("CLIP Retrieval Accuracy (Hit@1)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{save_dir}/clip_accuracy_comparison.png")
    plt.close()

    # 2. Summarizer Performance (ROUGE/BLEU)
    summ_df = load_history(experiments["Summarizer (Task 1)"])
    if summ_df is not None:
        plt.figure(figsize=(10, 6))
        if 'rougeL' in summ_df.columns:
            plt.plot(summ_df['epoch'], summ_df['rougeL'], marker='s', label="ROUGE-L")
        if 'bleu' in summ_df.columns:
            plt.plot(summ_df['epoch'], summ_df['bleu'], marker='^', label="BLEU")
        
        plt.title("Summarizer Performance")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(f"{save_dir}/summarizer_performance.png")
        plt.close()

    print(f"✅ Visualizations saved to {save_dir}")

if __name__ == "__main__":
    main()