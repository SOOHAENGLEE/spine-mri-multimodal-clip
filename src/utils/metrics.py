# src/utils/metrics.py
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_text_metrics(references, predictions):
    """
    텍스트 생성 평가 지표 (BLEU, ROUGE)
    references: List[str] (정답 문장들)
    predictions: List[str] (생성된 문장들)
    """
    smoothie = SmoothingFunction().method4
    # NLTK expects list of list of tokens for refs
    refs_tokenized = [[r.split()] for r in references]
    preds_tokenized = [p.split() for p in predictions]
    
    bleu_score = corpus_bleu(refs_tokenized, preds_tokenized, smoothing_function=smoothie)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    
    for ref, pred in zip(references, predictions):
        sc = scorer.score(ref, pred)
        r1.append(sc['rouge1'].fmeasure)
        r2.append(sc['rouge2'].fmeasure)
        rl.append(sc['rougeL'].fmeasure)
        
    return {
        "bleu": bleu_score * 100,
        "rouge1": np.mean(r1) * 100,
        "rouge2": np.mean(r2) * 100,
        "rougeL": np.mean(rl) * 100
    }

def calculate_retrieval_metrics(logits, targets):
    """
    [핵심 추가] 검색 성능 평가 지표 (Hit@K, MRR)
    logits: (Batch, Num_Candidates) - 유사도 점수
    targets: (Batch) - 정답 인덱스
    """
    # 1. 점수 기준으로 내림차순 정렬 (높은 점수가 상위 랭크)
    # sorted_indices: (Batch, Num_Candidates)
    _, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # 2. 정답 위치(Rank) 찾기
    # targets를 (Batch, 1)로 만들어서 브로드캐스팅 비교
    hits = (sorted_indices == targets.view(-1, 1))
    
    # hits.nonzero()는 (sample_idx, rank_idx) 튜플을 반환
    # 우리는 각 샘플별 정답이 몇 번째 랭크인지(rank_idx)가 필요함
    # +1을 해주는 이유는 인덱스가 0부터 시작하기 때문 (1등이 0번 인덱스)
    ranks = hits.nonzero(as_tuple=False)[:, 1] + 1
    ranks = ranks.float()
    
    batch_size = logits.size(0)
    
    # 3. Metrics 계산
    # Hit@K: 정답이 K등 안에 든 비율
    hit1 = (ranks <= 1).sum().item() / batch_size
    hit5 = (ranks <= 5).sum().item() / batch_size
    hit10 = (ranks <= 10).sum().item() / batch_size
    
    # MRR (Mean Reciprocal Rank): 1/등수 의 평균
    mrr = (1.0 / ranks).mean().item()
    
    return {
        "hit1": hit1,
        "hit5": hit5,
        "hit10": hit10,
        "mrr": mrr
    }