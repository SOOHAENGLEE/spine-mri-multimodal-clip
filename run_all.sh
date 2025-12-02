#!/bin/bash

# ==============================================================================
#  Spine MRI Multimodal Project - Perfect Parallel Pipeline
#  Optimized for Multi-GPU (A5000 x 4)
# ==============================================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# [DEADLOCK & ERROR PREVENTION]
# 1. Tokenizer 병렬화 끄기 (DataLoader와 충돌 방지)
export TOKENIZERS_PARALLELISM=false
# 2. CPU 스레드 충돌 방지
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CV_NUM_THREADS=0
# 3. DDP 통신 타임아웃 방지
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1

# [Accelerate Config]
# FP16 사용, 4 GPU
ACC_ARGS="--multi_gpu --num_processes=4 --gpu_ids=all --mixed_precision=fp16"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"

log() {
    echo -e "[$(date +'%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================================="
log "🚀 Spine MRI Pipeline (Full GPU Optimized)"
log "=========================================================="

# 1. Preprocessing (Tensor Caching 포함)
log ""
log ">>> [Step 1/6] Data Preprocessing & Caching"
# 데이터가 이미 있어도, Tensor Cache가 있는지 확실하지 않으므로 실행 권장 (이미 있으면 빠름)
python scripts/01_run_preprocess.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 2. Summarizer Training
log ""
log ">>> [Step 2/6] Training Summarizer"
accelerate launch $ACC_ARGS scripts/02_train_summarizer.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 3. Pseudo-label Generation (Accelerate)
log ""
log ">>> [Step 3/6] Generating Pseudo-labels"
accelerate launch $ACC_ARGS scripts/03_gen_pseudo_labels.py --config configs/model_config.yaml 2>&1 | tee -a "$LOG_FILE"

# 4. CLIP Training (Summary Mode) - Global Loss 적용
log ""
log ">>> [Step 4/6] Training CLIP (Summary Mode)"
accelerate launch $ACC_ARGS scripts/04_train_clip.py --config configs/model_config.yaml --mode summary 2>&1 | tee -a "$LOG_FILE"

# 5. CLIP Training (Full Report Mode) - Global Loss 적용
log ""
log ">>> [Step 5/6] Training CLIP (Full Report Mode)"
accelerate launch $ACC_ARGS scripts/04_train_clip.py --config configs/model_config.yaml --mode full_report 2>&1 | tee -a "$LOG_FILE"

# 6. Visualization
log ""
log ">>> [Step 6/6] Final Visualization"
python scripts/05_visualize_results.py 2>&1 | tee -a "$LOG_FILE"
# 시각화도 Accelerate로 빠르게
accelerate launch $ACC_ARGS scripts/06_comprehensive_viz.py 2>&1 | tee -a "$LOG_FILE"

log "🎉 Pipeline Finished Successfully!"