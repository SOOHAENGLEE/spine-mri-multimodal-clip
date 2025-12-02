# scripts/03_gen_pseudo_labels.py
import sys
import os
import torch
import json
import argparse
import yaml
import math
from transformers import AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from src.models.summarizer import ReportSummarizer

def main(args):
    # 1. Config & Accelerator
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Accelerate 초기화 (Mixed Precision 적용)
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # 경로 설정
    MODEL_PATH = f"{cfg['paths']['result_dir']}/summarizer/best_model.pt"
    UNLABELED_PATH = f"{cfg['paths']['output_dir']}/unlabeled.json"
    # 각 GPU 프로세스가 저장할 임시 파일 경로
    TEMP_OUTPUT_PATH = f"{cfg['paths']['output_dir']}/temp_aug_part_{accelerator.process_index}.json"
    FINAL_OUTPUT_PATH = f"{cfg['paths']['output_dir']}/augmented.json"
    LABELED_PATH = f"{cfg['paths']['output_dir']}/labeled.json"
    
    MODEL_NAME = cfg['models']['summarizer']['name']
    NUM_RETURN_SEQUENCES = 3
    BATCH_SIZE = 16 # GPU당 배치 사이즈

    if accelerator.is_main_process:
        print(f"🚀 Generating Pseudo-labels with {accelerator.num_processes} GPUs")
        if not os.path.exists(UNLABELED_PATH):
            print(f"❌ {UNLABELED_PATH} not found.")
            return

    # 2. 데이터 로드 및 분할 (Sharding)
    with open(UNLABELED_PATH, 'r') as f:
        all_unlabeled = json.load(f)
    
    # 전체 데이터를 GPU 개수만큼 나눔
    total_size = len(all_unlabeled)
    chunk_size = math.ceil(total_size / accelerator.num_processes)
    start_idx = accelerator.process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_size)
    
    my_chunk = all_unlabeled[start_idx:end_idx]
    
    print(f"   [GPU {accelerator.process_index}] Processing {len(my_chunk)} samples ({start_idx}~{end_idx})")

    # 3. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ReportSummarizer(MODEL_NAME)
    
    # 가중치 로드 (Map location 주의)
    if os.path.exists(MODEL_PATH):
        # DDP 학습된 체크포인트는 'module.' prefix가 있을 수 있음
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    
    # 모델을 Accelerator로 준비 (Model만 prepare 해도 됨, DataLoader를 안쓰고 리스트 순회하므로)
    model = model.to(device)
    model.eval()

    # 4. 추론 (Batch Processing)
    augmented_data = []
    
    # 내 청크 안에서 배치 처리
    for i in tqdm(range(0, len(my_chunk), BATCH_SIZE), desc=f"GPU {accelerator.process_index}", disable=not accelerator.is_local_main_process):
        batch_items = my_chunk[i : i+BATCH_SIZE]
        texts = [item['full_report'] for item in batch_items]
        
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                max_length=128,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=3
            )
            
        summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        
        # 결과 매핑
        for j, item in enumerate(batch_items):
            current_sums = summaries[j*NUM_RETURN_SEQUENCES : (j+1)*NUM_RETURN_SEQUENCES]
            for k, summ in enumerate(current_sums):
                new_item = item.copy()
                new_item['id'] = f"{item['id']}_aug{k}"
                new_item['summary'] = summ
                augmented_data.append(new_item)

    # 5. 부분 결과 저장
    with open(TEMP_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    # 모든 프로세스가 파일을 쓸 때까지 대기
    accelerator.wait_for_everyone()

    # 6. 결과 병합 (Main Process Only)
    if accelerator.is_main_process:
        print("🔄 Merging results from all GPUs...")
        final_augmented = []
        for i in range(accelerator.num_processes):
            part_path = f"{cfg['paths']['output_dir']}/temp_aug_part_{i}.json"
            if os.path.exists(part_path):
                with open(part_path, 'r') as f:
                    final_augmented.extend(json.load(f))
                os.remove(part_path) # 임시 파일 삭제
        
        # Labeled 데이터와 합치기
        if os.path.exists(LABELED_PATH):
            with open(LABELED_PATH, 'r') as f:
                labeled_data = json.load(f)
            final_dataset = labeled_data + final_augmented
        else:
            final_dataset = final_augmented
            
        with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Created {FINAL_OUTPUT_PATH} with {len(final_dataset)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args)