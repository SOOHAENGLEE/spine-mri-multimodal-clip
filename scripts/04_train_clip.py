import sys
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

# 데드락 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.data.datasets import CLIPDataset
from src.models.clip import SpineCLIP
from src.utils.logger import ExperimentLogger
from src.utils.metrics import calculate_retrieval_metrics

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    TASK_MODE = args.mode
    SAVE_DIR = f"{cfg['paths']['result_dir']}/clip_{TASK_MODE}"
    
    # [DDP 설정] find_unused_parameters=True는 이제 필요 없거나 False가 더 안전할 수 있음
    # 하지만 안전을 위해 켜두되, 모델 Forward 수정으로 해결
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="all", project_dir=SAVE_DIR, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    if accelerator.is_main_process:
        logger = ExperimentLogger(SAVE_DIR)
        print(f"🚀 Training CLIP ({TASK_MODE}) on {accelerator.num_processes} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(cfg['models']['clip']['text_encoder'])
    data_path = f"{cfg['paths']['output_dir']}/augmented.json"
    if not os.path.exists(data_path): data_path = f"{cfg['paths']['output_dir']}/labeled.json"

    ds = CLIPDataset(data_path, tokenizer, img_size=tuple(cfg['train']['img_size']), text_col=TASK_MODE, is_train=True)
    train_size = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)
    
    model = SpineCLIP(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    best_mrr = 0.0

    for epoch in range(cfg['train']['epochs']):
        model.train()
        train_loss = 0
        
        iterator = tqdm(train_loader, desc=f"Ep {epoch+1}", disable=not accelerator.is_main_process)
        
        for batch in iterator:
            imgs = batch['images']
            ids = batch['input_ids']
            mask = batch['attention_mask']
            
            # Forward (이제 scale도 리턴받음)
            img_emb, txt_emb, scale = model(imgs, ids, mask)
            
            # [Safe Global Loss Implementation]
            # 1. Gather all embeddings (Gradient 미포함)
            # accelerator.gather()는 detach된 텐서를 모아줍니다.
            all_img = accelerator.gather(img_emb)
            all_txt = accelerator.gather(txt_emb)
            
            # 2. 내 데이터(Local)가 전체(Global)에서 어디에 위치하는지 찾기
            # (Gather된 텐서는 [GPU0, GPU1, GPU2, GPU3] 순서로 붙어있음)
            
            # 3. 하지만 미분이 끊기면 학습이 안 되므로, 
            # "Local Image vs Global Text" + "Local Text vs Global Image" 로 계산해야 함.
            # 이러면 Local 부분에는 Gradient가 흐르고, Global 부분(남의 것)은 Negative Sample 역할만 함.
            
            # Matmul: (Local_B, D) @ (Global_B, D).T -> (Local_B, Global_B)
            logits_per_image = scale * img_emb @ all_txt.t()
            logits_per_text = scale * txt_emb @ all_img.t()
            
            # 정답 라벨 (Global 내에서의 내 위치)
            # rank 0: 0~15, rank 1: 16~31 ...
            local_batch_size = img_emb.size(0)
            global_offset = accelerator.process_index * local_batch_size
            labels = torch.arange(local_batch_size, device=device) + global_offset
            
            loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        metrics_sum = {'hit1':0, 'mrr':0}
        
        for batch in val_loader:
            with torch.no_grad():
                # Val에서는 scale 그냥 접근해도 됨 (Backward 안하니까)
                # 하지만 model forward 형식이 바뀌었으므로 맞춰줌
                img_emb, txt_emb, scale = model(batch['images'], batch['input_ids'], batch['attention_mask'])
                
                logits = scale * img_emb @ txt_emb.t()
                labels = torch.arange(len(img_emb), device=device)
                
                metrics = calculate_retrieval_metrics(logits, labels)
                metrics_sum['hit1'] += metrics['hit1']
                metrics_sum['mrr'] += metrics['mrr']

        if accelerator.is_main_process:
            avg_loss = train_loss / len(train_loader)
            avg_hit1 = metrics_sum['hit1'] / len(val_loader)
            print(f"Ep {epoch+1} | Loss: {avg_loss:.4f} | Val Hit@1: {avg_hit1:.3f}")
            logger.log(epoch+1, {'train_loss': avg_loss, 'val_acc': avg_hit1})
            
            if avg_hit1 > best_mrr:
                best_mrr = avg_hit1
                torch.save(accelerator.unwrap_model(model).state_dict(), f"{SAVE_DIR}/best_model.pt")

    if accelerator.is_main_process: logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--mode", type=str, default="summary")
    args = parser.parse_args()
    main(args)