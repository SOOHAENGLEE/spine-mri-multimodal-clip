# src/utils/logger.py
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, "history.json")
        self.history = []

    def log(self, epoch, metrics):
        entry = {'epoch': epoch}
        entry.update(metrics)
        self.history.append(entry)
        self.save()

    def save(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def plot(self):
        if not self.history: return
        df = pd.DataFrame(self.history)
        
        # 1. Loss Plot
        plt.figure(figsize=(10, 5))
        if 'train_loss' in df.columns:
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        if 'val_loss' in df.columns:
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

        # 2. ROUGE/BLEU Plot (Summarizer Task)
        if 'bleu' in df.columns or 'rougeL' in df.columns:
            plt.figure(figsize=(10, 5))
            if 'bleu' in df.columns:
                plt.plot(df['epoch'], df['bleu'], label='BLEU', marker='s')
            if 'rougeL' in df.columns:
                plt.plot(df['epoch'], df['rougeL'], label='ROUGE-L', marker='^')
            plt.title('Text Generation Metrics')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "text_metrics.png"))
            plt.close()

        # 3. Accuracy Plot (CLIP Task)
        if 'val_acc' in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df['epoch'], df['val_acc'], label='Acc@1', marker='o')
            if 'val_acc5' in df.columns:
                plt.plot(df['epoch'], df['val_acc5'], label='Acc@5', marker='x')
            plt.title('Retrieval Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "accuracy_curve.png"))
            plt.close()