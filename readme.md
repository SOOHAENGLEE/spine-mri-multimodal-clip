# Spine MRI 2.5D-Text CLIP Model for Study-level Report Summary Matching

[](https://pytorch.org/)
[](https://huggingface.co/)
[](https://github.com/huggingface/accelerate)

본 프로젝트는 **척추(Spine) MRI의 2.5D 시퀀스 데이터**와 **영상의학 판독문(Radiology Report)** 간의 의미적 정렬(Semantic Alignment)을 수행하는 멀티모달 AI 연구 프로젝트입니다.
복잡한 3D 볼륨 정보를 효율적으로 처리하기 위해 **2.5D Stacking** 기법을 적용하고, 비정형 텍스트의 노이즈를 줄이기 위해 **BioBART 기반 요약(Summarization)** 모델을 결합한 파이프라인을 제안합니다.

-----

## 📌 1. Background & Motivation

척추 MRI는 단일 슬라이스(Slice)만으로는 병변의 유무와 위치를 정확히 파악하기 어렵습니다. 인접한 슬라이스 간의 맥락(Context)이 압박 골절, 종양, 디스크 탈출 등의 진단에 필수적이기 때문입니다. 또한, 임상 현장의 판독문은 비정형 텍스트로 작성되어 영상과 1:1로 매칭되지 않는 문제가 있습니다.

기존 2D CNN 기반 연구나 단일 슬라이스 처리 방식은 이러한 3차원적 맥락을 놓칠 위험이 큽니다. 본 프로젝트는 이를 해결하기 위해 다음 두 가지 핵심 전략을 사용합니다:

1.  **2.5D Representation:** 다수의 시상면(Sagittal) 슬라이스를 채널 축으로 쌓아(Stacking) 3D 정보를 보존하면서도 효율적인 2D 백본을 활용합니다.
2.  **Text Summarization:** 긴 판독문을 핵심 진단명 위주로 요약하여 CLIP 학습의 노이즈를 제거하고 효율성을 증대시킵니다.

-----

## ⚙️ 2. Methodology

본 모델은 **Contrastive Language-Image Pre-training (CLIP)** 프레임워크를 기반으로 하며, 의료 영상 특성에 맞춰 인코더 구조와 손실 함수를 최적화하였습니다.

### 2.1. 2.5D Image Encoding Strategy

MRI 데이터는 환자마다 슬라이스 개수가 다르므로, 고정된 깊이 $D$ (본 연구에서는 $D=15$)로 정규화합니다. 입력 이미지 $X_{img}$는 다음과 같이 정의됩니다.

$$X_{img} \in \mathbb{R}^{D \times H \times W}$$

여기서 $D=15, H=224, W=224$입니다. 이를 처리하기 위해 **ConvNeXt-Base** 모델의 첫 번째 Convolution Layer를 수정하여 $D$ 채널 입력을 받을 수 있도록 확장하였습니다.

이미지 임베딩 $z_{img}$는 Global Average Pooling (GAP)과 선형 변환(Linear Projection)을 통해 얻어집니다.

$$f_{img} = \text{GAP}(\text{ConvNeXt}(X_{img})) \in \mathbb{R}^{1024}$$
$$z_{img} = W_{img} \cdot f_{img}, \quad W_{img} \in \mathbb{R}^{512 \times 1024}$$
$$\hat{z}_{img} = \frac{z_{img}}{\|z_{img}\|_2}$$

### 2.2. Text Encoding with Summarization

원본 판독문 $T_{raw}$는 불필요한 서술이나 아티팩트 정보를 포함하고 있어 노이즈가 많습니다. 이를 해결하기 위해 **BioBART** 기반 요약 모델을 사용하여 핵심 소견 $T_{summary}$를 추출합니다.

이후, **Multilingual BERT**를 텍스트 인코더로 사용하여 [CLS] 토큰의 출력을 텍스트 임베딩으로 변환합니다.

$$H_{txt} = \text{BERT}(T_{summary})$$
$$z_{txt} = W_{txt} \cdot H_{txt}^{\text{[CLS]}}, \quad W_{txt} \in \mathbb{R}^{512 \times 768}$$
$$\hat{z}_{txt} = \frac{z_{txt}}{\|z_{txt}\|_2}$$

### 2.3. Contrastive Objective Formulation

모델은 배치(Batch) 내 $N$개의 이미지-텍스트 쌍 $\{(I_i, T_i)\}_{i=1}^N$에 대해, 대응되는 쌍(Positive Pair)의 유사도는 최대화하고, 그렇지 않은 쌍(Negative Pair)의 유사도는 최소화하도록 학습됩니다.

두 정규화된 임베딩 간의 코사인 유사도 행렬(Logits) $S$는 다음과 같습니다.

$$S_{ij} = \hat{z}_{img, i} \cdot \hat{z}_{txt, j}^\top$$

최종 Loss Function은 학습 가능한 온도 매개변수 $\tau$ (Temperature parameter)를 적용한 **Symmetric Cross-Entropy Loss**를 사용합니다.

1.  **Image-to-Text Loss:**
    $$\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii} / \tau)}{\sum_{j=1}^{N} \exp(S_{ij} / \tau)}$$

2.  **Text-to-Image Loss:**
    $$\mathcal{L}_{T \to I} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii} / \tau)}{\sum_{j=1}^{N} \exp(S_{ji} / \tau)}$$

3.  **Total Loss:**
    $$\mathcal{L}_{total} = \frac{1}{2} (\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I})$$

-----

## 📊 3. Experimental Results

본 프로젝트는 가톨릭중앙의료원 산하 6개 병원의 척추 MRI 데이터셋(892명 환자)을 사용하여 실험을 진행했습니다.

### 3.1. Summarization Performance (Task 1)

판독문 요약 모델(`BioBART`)의 학습 경과에 따른 성능 변화입니다.

  * **ROUGE-L:** 학습이 진행됨에 따라 **약 70점** 대에 도달하며, 생성된 요약문이 정답 레이블과 구조적으로 매우 유사함을 확인했습니다.
  * **BLEU:** 약 **40점** 대를 기록하여, 의학적 핵심 용어들이 정확하게 포함되고 있음을 보여줍니다.

### 3.2. CLIP Retrieval Accuracy (Task 2)

제안한 방법(Summary 기반 학습)과 베이스라인(Full Report 기반 학습)의 검색 정확도 비교입니다.

  * **Proposed (Summary Mode):** 핵심 정보만 요약된 텍스트로 학습했을 때, **Hit@1 Accuracy가 99% 이상**에 빠르게 도달하며 매우 안정적인 성능을 보였습니다. 이는 텍스트의 노이즈 제거가 멀티모달 학습 효율에 결정적임을 시사합니다.
  * **Baseline (Full Report):** 긴 원본 텍스트를 사용한 경우, 수렴 속도가 상대적으로 느리지만 최종적으로는 유사한 성능에 도달했습니다.

### 3.3. Qualitative Analysis (Retrieval Examples)

실제 MRI 영상(Query)에 대해 모델이 가장 유사하다고 판단한 텍스트를 검색한 결과입니다.

  * **Query Image:** 환자의 2.5D MRI (중간 슬라이스 시각화).
  * **Ground Truth Summary:** 해당 환자의 실제 요약 판독문.
  * **Result:** 모델은 척추의 **압박 골절(Compression fracture), 디스크 탈출(Disc bulge), 척추체 변형** 등의 미세한 병변을 정확하게 인식하고, 이에 해당하는 텍스트 설명과 매칭시키는 데 성공했습니다.

-----

## 📂 4. Project Structure

```bash
.
├── configs/                # 학습 설정 파일 (Hyperparameters)
│   └── model_config.yaml   
├── data/
│   ├── raw/                # 원본 데이터
│   └── processed/          # 전처리된 .pt 텐서 및 JSON 라벨
├── results/                # 학습 결과 및 시각화 저장소
│   ├── clip_summary/       # CLIP 모델 가중치
│   ├── summarizer/         # 요약 모델 가중치
│   ├── comparison/         # 결과 그래프 (Performance Plots)
│   └── viz_retrieval.png   # 검색 결과 시각화
├── scripts/                # 실행 스크립트 모음
│   ├── 01_run_preprocess.py    # 데이터 전처리 및 캐싱
│   ├── 02_train_summarizer.py  # 요약 모델 학습
│   ├── 03_gen_pseudo_labels.py # Pseudo-label 생성
│   ├── 04_train_clip.py        # CLIP 모델 학습
│   ├── 05_visualize_results.py # 결과 그래프 그리기
│   └── 06_comprehensive_viz.py # 심층 시각화
├── src/                    # 소스 코드 (모델, 데이터셋 정의)
├── run_all.sh              # 전체 파이프라인 자동 실행 스크립트
└── requirements.txt        # 의존성 패키지
```

-----

## 🚀 5. Usage

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install opencv-python-headless  # For server environments
```

### Running Pipeline

전체 파이프라인(전처리 → 요약 학습 → 데이터 생성 → CLIP 학습 → 시각화)을 한 번에 실행합니다.
(A5000 x 4 Multi-GPU 환경에 최적화되어 있습니다.)

```bash
chmod +x run_all.sh
./run_all.sh
```

-----

## 👥 Team Info

**Natural Language Processing Team 1**

  * **Leader:** Jo Jun-hee
  * **Members:** Lee Jun, Lee Su-haeng, Park Sang-yu