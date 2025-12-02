# scripts/01_run_preprocess.py
import sys
import os
import argparse

# [중요] 프로젝트 루트 경로를 sys.path에 추가하여 src 모듈을 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts 폴더
project_root = os.path.dirname(current_dir)              # 프로젝트 최상위 폴더
sys.path.append(project_root)

from src.data.preprocess import run_preprocessing

if __name__ == "__main__":
    # [수정] argparse를 사용하여 실행 인자(--config)를 받도록 변경
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config_path = args.config
    
    # 경로가 상대경로라면 프로젝트 루트 기준으로 절대경로로 변환 (안전장치)
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        exit(1)
        
    print(f"⚙️  Using config: {config_path}")
    run_preprocessing(config_path)