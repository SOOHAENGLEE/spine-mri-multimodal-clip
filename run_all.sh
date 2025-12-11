#!/bin/bash

# ==============================================================================
#  Spine MRI Multimodal Project - Full Pipeline (FINAL)
# ==============================================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# [DEADLOCK & ERROR PREVENTION]
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CV_NUM_THREADS=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1

# [Accelerate Config]
# FP16 사용, 3 GPU (A6000 환경 최적화)
ACC_ARGS="--multi_gpu --num_processes=3 --gpu_ids=all --mixed_precision=fp16"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"

log() {
    echo -e "[$(date +'%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 🚀 [CLEANUP] 시작 전 찌꺼기 파일 정리
find "$PROJECT_ROOT" -maxdepth 2 -type f \( -name 'pymp-*' -o -name 'tmp-*' -o -name '*-mp' -o -name '.-mp' \) -delete 2>/dev/null || true
rm -rf "$PROJECT_ROOT"/torchelastic_* 2>/dev/null || true

log "=========================================================="
log "🚀 Spine MRI Multimodal Pipeline (A6000 x 3 Optimized)"
log "=========================================================="

# 1. Data Preprocessing & Caching
log ""
log ">>> [Step 1/10] Data Preprocessing & Caching"
python scripts/01_run_preprocess.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 2. Text-to-Text Summarizer Training (Task 1. 요약)
log ""
log ">>> [Step 2/10] Training Text-to-Text Summarizer (Full Report -> Summary)"
accelerate launch $ACC_ARGS scripts/02_train_summarizer.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 3. Pseudo-label Generation (CLIP/Generator 학습을 위한 데이터 증강)
log ""
log ">>> [Step 3/10] Generating Pseudo-labels (Data Augmentation) - (임시 파일은 data/processed/temp_pseudo_parts에 저장 후 삭제)"
accelerate launch $ACC_ARGS scripts/03_gen_pseudo_labels.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 4. CLIP Training (Summary Mode) (Task 2. CLIP 실험 1/2) - Generator의 Image Encoder Pre-training 역할
log ""
log ">>> [Step 4/10] Training CLIP (Summary Mode) - Pre-training for Generator"
accelerate launch $ACC_ARGS scripts/04_train_clip.py --config configs/model_config.yaml --mode summary 2>&1 | tee -a "$LOG_FILE"

# 5. CLIP Training (Full Report Mode) (Task 2. CLIP 실험 2/2)
log ""
log ">>> [Step 5/10] Training CLIP (Full Report Mode)"
accelerate launch $ACC_ARGS scripts/04_train_clip.py --config configs/model_config.yaml --mode full_report 2>&1 | tee -a "$LOG_FILE"

# 6. Image-to-Text Generator Training (Task 3. 2.5D MRI -> 요약문 생성) - 🔥 CLIP 가중치 전이 학습 사용
log ""
log ">>> [Step 6/10] Training Image-to-Summary Generator (2.5D MRI -> Summary)"
accelerate launch $ACC_ARGS scripts/08_train_generator.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"


# 7. Basic Visualization (Summary/CLIP 비교 그래프)
log ""
log ">>> [Step 7/10] Basic Visualization"
python scripts/05_visualize_results.py 2>&1 | tee -a "$LOG_FILE"

# 8. Comprehensive CLIP Visualization (Heatmap, Projection, Retrieval Demo)
log ""
log ">>> [Step 8/10] Comprehensive CLIP Visualization"
accelerate launch $ACC_ARGS scripts/06_comprehensive_viz.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 9. Generator Output Visualization (NEW)
log ""
log ">>> [Step 9/10] Image-to-Summary Generation Visualization"
accelerate launch $ACC_ARGS scripts/09_viz_generator_demo.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 10. Final Cleanup - 찌꺼기 파일 정리 (강화된 로직)
log ""
log ">>> [Step 10/10] Final Cleanup: Removing temporary distributed/multiprocessing files..."
# 🔥 [강화된 로직] 현재 디렉토리 및 숨김 파일을 포함하여 찌꺼기 파일 정리
find "$PROJECT_ROOT" -maxdepth 2 -type f \( -name 'pymp-*' -o -name 'tmp-*' -o -name '*-mp' -o -name '.-mp' \) -delete
rm -rf "$PROJECT_ROOT"/torchelastic_* 2>/dev/null || true
rm -f "$PROJECT_ROOT"/.accelerate_state* 2>/dev/null || true

log "🎉 Pipeline Finished Successfully!"