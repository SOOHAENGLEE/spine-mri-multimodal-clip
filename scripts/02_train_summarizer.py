# scripts/02_train_summarizer.py
import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.data.datasets import SummarizationDataset
from src.models.summarizer import ReportSummarizer
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_text_metrics

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    SAVE_DIR = cfg['paths']['result_dir'] + "/summarizer"
    MODEL_NAME = cfg['models']['summarizer']['name']
    
    accelerator = Accelerator(log_with="all", project_dir=SAVE_DIR)
    device = accelerator.device
    
    if accelerator.is_main_process:
        logger = ExperimentLogger(SAVE_DIR)
        print(f"🚀 Training Summarizer (Accelerate) | Model: {MODEL_NAME}")

    BATCH_SIZE = cfg['train']['batch_size']
    EPOCHS = cfg['train']['epochs']
    LR = float(cfg['train']['lr'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_ds = SummarizationDataset(cfg['paths']['output_dir'] + "/labeled.json", tokenizer)
    
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    # [설정] 효율적인 데이터 로더
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ReportSummarizer(MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val_loss = float('inf') # Loss 기준 베스트 저장

    for epoch in range(EPOCHS):
        # 1. Train Loop
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Ep {epoch+1} Train", disable=not accelerator.is_main_process)
        
        for batch in progress_bar:
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                labels=batch['labels']
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 2. Validation (Loss Only) - 아주 빠름!
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels']
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 3. Conditional Generation (조건부 생성 평가) - 느리니까 필요할 때만!
        # 조건: Loss가 기존보다 좋거나 OR 5 에포크마다 확인
        is_best = avg_val_loss < best_val_loss
        should_eval_metric = is_best or ((epoch + 1) % 5 == 0)
        
        metrics = {'rougeL': 0.0, 'bleu': 0.0} # 기본값

        if should_eval_metric:
            if accelerator.is_main_process:
                print(f"✨ Validation Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Running Generation...")
            
            decoded_preds = []
            decoded_labels = []
            
            # 여기가 오래 걸리는 구간 (생성)
            for batch in tqdm(val_loader, desc="   -> Generating...", disable=not accelerator.is_main_process):
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    gen_ids = unwrapped_model.generate(
                        batch['input_ids'], 
                        batch['attention_mask'], 
                        max_length=128, 
                        num_beams=4
                    )
                    # Gather
                    gen_ids = accelerator.pad_across_processes(gen_ids, dim=1, pad_index=tokenizer.pad_token_id)
                    gen_ids = accelerator.gather(gen_ids)
                    labels = accelerator.pad_across_processes(batch['labels'], dim=1, pad_index=tokenizer.pad_token_id)
                    labels = accelerator.gather(labels)

                    if accelerator.is_main_process:
                        preds = tokenizer.batch_decode(gen_ids.cpu(), skip_special_tokens=True)
                        targets = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)
                        decoded_preds.extend(preds)
                        decoded_labels.extend(targets)
            
            if accelerator.is_main_process:
                metrics = compute_text_metrics(decoded_labels, decoded_preds)

        # 4. 저장 및 로깅
        if accelerator.is_main_process:
            print(f"Ep {epoch+1} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if should_eval_metric:
                print(f"   -> ROUGE-L: {metrics['rougeL']:.2f} | BLEU: {metrics['bleu']:.2f}")
            else:
                print(f"   -> (Skipped Generation to save time)")

            logger.log(epoch+1, {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'bleu': metrics['bleu'],
                'rougeL': metrics['rougeL']
            })
            
            # Loss가 개선되었으면 모델 저장
            if is_best:
                best_val_loss = avg_val_loss
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), f"{SAVE_DIR}/best_model.pt")
                print(f"   🔥 New Best Model Saved (by Loss)!")

    if accelerator.is_main_process:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args)