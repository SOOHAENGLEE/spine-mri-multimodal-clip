import os
import json
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import torchvision.transforms as T
from src.utils.common import load_config

def clean_text_for_input(text):
    text = str(text)
    text = re.sub(r'Clinical Information:.*', '', text)
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_id(s):
    return str(s).strip().upper()

def build_global_index(root_path):
    print(f"🏗 Indexing ALL folders under: {root_path}")
    path_index = defaultdict(list)
    for root, dirs, files in os.walk(root_path):
        for d in dirs:
            if 'R' in d.upper():
                norm_name = re.sub(r'[^A-Z0-9]', '', d.upper())
                path_index[norm_name].append(os.path.join(root, d))
    return path_index

def load_and_cache_images(folder_path, save_path, img_size=(320, 320)):
    """이미지 15장을 로드하여 Tensor로 변환 후 .pt로 저장 (캐싱)"""
    p = Path(folder_path)
    files = sorted(list(p.glob("*.jpg")))
    
    if not files: return False

    # 15장 샘플링 (보간법)
    if len(files) < 15:
        indices = np.linspace(0, len(files)-1, 15, dtype=int)
        selected = [files[i] for i in indices]
    elif len(files) == 15:
        selected = files
    elif len(files) == 17:
        selected = files[1:-1]
    else:
        indices = np.linspace(0, len(files)-1, 15, dtype=int)
        selected = [files[i] for i in indices]

    # Transform (Resize & Normalize)
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    imgs = []
    for f in selected:
        try:
            img = Image.open(f).convert("L")
            imgs.append(transform(img))
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")
            imgs.append(torch.zeros(1, *img_size))

    # (15, H, W) Tensor로 병합
    tensor_stack = torch.cat(imgs, dim=0)
    # 압축 없이 빠르게 저장
    torch.save(tensor_stack, save_path)
    return True

def find_best_sequence(candidate_paths, min_count=5):
    best_path = None
    max_score = -1
    for study_path in candidate_paths:
        for root, dirs, files in os.walk(study_path):
            jpgs = [f for f in files if f.lower().endswith('.jpg')]
            if len(jpgs) < min_count: continue
            score = 0
            path_lower = root.lower()
            if 'sag' in path_lower: score += 1000
            if 't2' in path_lower: score += 500
            score += len(jpgs)
            if score > max_score:
                max_score = score
                best_path = root
    return best_path

def run_preprocessing(config_path):
    cfg = load_config(config_path)
    
    # 데이터 로드 (인코딩 호환성)
    try: df = pd.read_csv(cfg['paths']['raw_csv'], encoding='cp949')
    except: df = pd.read_csv(cfg['paths']['raw_csv'], encoding='utf-8')
    
    folder_index = build_global_index(cfg['paths']['image_root'])
    
    # 캐시 폴더 생성
    cache_dir = Path(cfg['paths']['output_dir']) / "tensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    labeled, unlabeled = [], []
    stats = defaultdict(int)
    
    print("🔍 Matching and Caching Images (Creating .pt files)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        s_id = str(row[cfg['columns']['study_col']]).strip()
        s_id_norm = re.sub(r'[^A-Z0-9]', '', s_id.upper())
        
        candidates = []
        for key in folder_index:
            if s_id_norm in key: candidates.extend(folder_index[key])
        
        if not candidates:
            stats['fail'] += 1
            continue
            
        best_path = find_best_sequence(candidates, cfg['settings']['min_slice_count'])
        if not best_path:
            stats['no_img'] += 1
            continue
            
        # [핵심] 이미지를 .pt로 저장 (이미 있으면 건너뜀)
        tensor_save_path = cache_dir / f"{s_id}_{idx}.pt"
        if not tensor_save_path.exists():
            success = load_and_cache_images(best_path, tensor_save_path, tuple(cfg['train']['img_size']))
            if not success:
                stats['error'] += 1
                continue
        
        stats['success'] += 1
        
        full_text = str(row[cfg['columns']['full_report_col']]).strip()
        label = str(row[cfg['columns']['summary_label_col']]).strip()
        
        item = {
            "id": f"{s_id}_{idx}",
            "image_path": str(tensor_save_path), # .pt 경로 저장
            "full_report": clean_text_for_input(full_text)
        }
        
        if label != "" and label.lower() != "nan":
            item["summary"] = label
            labeled.append(item)
        else:
            item["summary"] = None
            unlabeled.append(item)
            
    with open(f"{cfg['paths']['output_dir']}/labeled.json", 'w') as f: json.dump(labeled, f, indent=2)
    with open(f"{cfg['paths']['output_dir']}/unlabeled.json", 'w') as f: json.dump(unlabeled, f, indent=2)
    
    print(f"✅ Preprocessing Done. Matched: {stats['success']} (Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)})")