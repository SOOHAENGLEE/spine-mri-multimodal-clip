# scripts/08_train_generator.py 파일 내용 전체

import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# 데드락 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.data.datasets import CLIPDataset
from src.models.generator import MRIReportGenerator
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_text_metrics

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    SAVE_DIR = cfg['paths']['result_dir'] + "/image_to_summary"
    MODEL_NAME = cfg['models']['summarizer']['name']
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="all", project_dir=SAVE_DIR, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    
    if accelerator.is_main_process:
        logger = ExperimentLogger(SAVE_DIR)
        print(f"🚀 Training MRI -> Summary Generator | Model: {MODEL_NAME}")

    BATCH_SIZE = cfg['train']['batch_size']
    EPOCHS = cfg['train']['epochs']
    LR = float(cfg['train']['lr'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ====================================================================
    # 🔥 데이터 경로 폴백 로직 (augmented.json -> labeled.json)
    # ====================================================================
    data_path = f"{cfg['paths']['output_dir']}/augmented.json"
    fallback_path = f"{cfg['paths']['output_dir']}/labeled.json"
    
    if not os.path.exists(data_path): 
        if os.path.exists(fallback_path):
             data_path = fallback_path
             if accelerator.is_main_process:
                 print(f"⚠️ Warning: augmented.json not found. Falling back to labeled.json.")
        else:
             raise FileNotFoundError(f"Required data file not found at {data_path} or {fallback_path}")
             
    full_ds = CLIPDataset(data_path, tokenizer, text_col='summary', is_train=True)
    
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # MRIReportGenerator 로드
    model = MRIReportGenerator(text_model_name=MODEL_NAME)
    
    # ====================================================================
    # 🔥 CLIP Image Encoder 가중치 전이 학습 로직
    # ====================================================================
    CLIP_CHECKPOINT_PATH = f"{cfg['paths']['result_dir']}/clip_summary/best_model.pt"
    
    if os.path.exists(CLIP_CHECKPOINT_PATH):
        if accelerator.is_main_process:
            print(f"🔄 Initializing Image Encoder with weights from CLIP: {CLIP_CHECKPOINT_PATH}")
        
        clip_state_dict = torch.load(CLIP_CHECKPOINT_PATH, map_location='cpu')
        new_clip_state = {k.replace('module.', ''): v for k, v in clip_state_dict.items()}
        
        clip_img_encoder_state = {}
        for k, v in new_clip_state.items():
            if k.startswith('img_enc.'):
                if 'global_proj' in k: 
                    continue 
                new_k = k.replace('img_enc.', '') 
                clip_img_encoder_state[new_k] = v

        missing_keys, unexpected_keys = model.img_encoder.load_state_dict(
            clip_img_encoder_state, 
            strict=False
        )
        
        if accelerator.is_main_process:
            print(f"✅ Image Encoder initialized with CLIP weights.")
            if 'proj.weight' in missing_keys and 'proj.bias' in missing_keys:
                print("   (Missing keys are the final linear layer 'proj', which is intended due to dimension mismatch.)")
            
    # ====================================================================
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val_loss = float('inf')

    # 🔥 [수정] BOS 토큰 ID를 미리 가져옵니다. (Generate용)
    BOS_TOKEN_ID = tokenizer.bos_token_id or tokenizer.cls_token_id

    for epoch in range(EPOCHS):
        # 1. Train Loop
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Ep {epoch+1} Train", disable=not accelerator.is_main_process)
        
        for batch in progress_bar:
            labels = batch['input_ids']
            
            outputs = model(
                images=batch['images'], 
                labels=labels
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 2. Validation (Loss Only)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['input_ids']
                outputs = model(images=batch['images'], labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 3. Conditional Generation (평가)
        is_best = avg_val_loss < best_val_loss
        should_eval_metric = is_best or ((epoch + 1) % 5 == 0)
        
        metrics = {'rougeL': 0.0, 'bleu': 0.0}

        if should_eval_metric:
            if accelerator.is_main_process:
                print(f"✨ Validation Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Running Generation...")
            
            decoded_preds = []
            decoded_labels = []
            
            for batch in tqdm(val_loader, desc="   -> Generating...", disable=not accelerator.is_main_process):
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    # 🔥 [수정] 더미 input_ids를 생성하여 generate에 전달합니다.
                    # 이를 통해 transformers가 encoder_outputs.last_hidden_state를 참조하지 않도록 우회합니다.
                    batch_size = batch['images'].size(0)
                    dummy_input_ids = torch.full(
                        (batch_size, 1), 
                        BOS_TOKEN_ID, 
                        dtype=torch.long, 
                        device=device
                    )
                    
                    gen_ids = unwrapped_model.generate(
                        batch['images'].to(device),
                        input_ids=dummy_input_ids, # 🔥 디코더의 시작 토큰을 명시적으로 전달
                        max_length=128, 
                        num_beams=4
                    )
                    
                    gen_ids = accelerator.pad_across_processes(gen_ids, dim=1, pad_index=tokenizer.pad_token_id)
                    gen_ids = accelerator.gather(gen_ids)
                    
                    labels = accelerator.pad_across_processes(batch['input_ids'], dim=1, pad_index=tokenizer.pad_token_id)
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

            logger.log(epoch+1, {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'bleu': metrics['bleu'],
                'rougeL': metrics['rougeL']
            })
            
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