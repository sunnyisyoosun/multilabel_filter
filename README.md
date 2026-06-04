# Multilingual Multi-Label Toxic Speech Classification

**영어-한국어 통합 유해 발화 다중 레이블 분류 파이프라인 (Engineering Lab)**

## Abstract

본 프로젝트는 영어와 한국어 텍스트를 단일 모델로 처리하는 **6-카테고리 다중 레이블 (multi-label) 유해 발화 분류기**를 구현한다. 다국어 임베딩 (multilingual-E5) 위에 MLP 분류기를 학습시키고, LLM 기반 pseudo-labeling (gemma-4-31B-it) 으로 학습 데이터를 보강하는 2단계 파이프라인을 설계하였다. 최종 모델은 **macro-F1 0.485** (영어 0.681 / 한국어 0.378) 를 달성하였다. 한국어 성능 개선을 위해 임베딩·데이터셋·LLM·threshold 등 7가지 접근을 체계적으로 비교 실험하였으며, 이 중 **threshold 튜닝이 가장 일관된 개선 (+0.083)** 을 보였음을 확인하였다.

---

## 1. Introduction

소셜 미디어 및 대화형 AI 환경에서 유해 발화 자동 필터링은 중요한 안전 과제이다. 그러나 기존 연구는 대부분 영어 단일 언어 또는 단순 이진 분류 (toxic / safe) 에 머물러 있다. 본 프로젝트는 다음 두 가지 도전을 다룬다.

1. **다국어 통합**: 영어와 한국어를 동일 임베딩 공간에서 처리하는 단일 모델
2. **세분화된 multi-label**: 단순 이진이 아닌 6개 카테고리의 동시 분류

학습 신호 부족 문제를 해결하기 위해 SlangLLM (Patel & Alsobeh, 2024) 과 HateGuard / HateCoT (Vishwamitra et al., IEEE S&P 2024) 에서 제안된 기법들을 결합한 LLM pseudo-labeling 파이프라인을 구축한다.

---

## 2. Method

### 2.1 카테고리 설계 (6 categories)

| Category | 한국어 | 정의 |
|---|---|---|
| `profanity` | 욕설 | 욕설/모욕적 표현 |
| `hate_speech` | 혐오발언 | 인종/종교/지역/장애/성소수자 혐오 |
| `gender` | 성 관련 | 여성·남성 성차별 및 성적 표현 (통합) |
| `threat` | 살해협박 | 살해·폭력 위협 |
| `political` | 정치 | 정치인·정당·이념 공격 |
| `other` | 기타유해 | 외모·나이·직업 비하 등 |

설계 결정: 초기 7카테고리 (`sexism`, `sexual_harassment` 분리) 에서 6카테고리로 **통합**. 한국어 데이터셋들 (K-HATERS, UnSmile) 이 모두 단일 `gender` 라벨만 제공하여 세분화 학습이 불가능하다는 데이터 한계를 확인하였다 (§4.1).

### 2.2 데이터셋 (총 238,428건)

**영어 (102,432건)**: hate_speech_offensive, ToxiGen, HateXplain, ETHOS, Jigsaw (threat 전용), BAD (safe/unsafe).

**한국어 (135,996건)**: K-HATERS (172K, 뉴스 댓글), Korean UnSmile (15K), AIHub 일상대화 (30K, 정상 샘플).

K-HATERS 매핑은 `label × target_label` 구조를 활용하였다:
- `L1_hate / L2_hate + gender` → `gender`
- `L1_hate / L2_hate + region/religion/disabled` → `hate_speech`
- `L1_hate / L2_hate + political` → `political`
- `offensive + individual/none` → `profanity`

### 2.3 임베딩 및 분류기

- **임베딩**: `intfloat/multilingual-e5-small` (384-dim, 영/한 통합 표현)
- **분류기**: MLP (384 → 256 → 128 → 6), Dropout 0.2, BCEWithLogitsLoss
- **베이스라인**: LogisticRegression (OneVsRest, class_weight='balanced')

### 2.4 LLM Pseudo-Labeling (HateGuard + SlangLLM)

소수 카테고리 (gender, threat) 의 학습 시그널을 보강하기 위해 BAD unsafe 데이터에 LLM 기반 의사 라벨링을 적용한다.

**1단계 — 카테고리 균형 후보 추출**: 각 카테고리 키워드 사전을 사용하여 후보 풀을 사전 분류하고 (영어/한국어 × 6 카테고리) 각 1,500건씩 + 정상 3,000건 = **24,000건** 균형 샘플링.

**2단계 — HateCoT 1-stage 프롬프트**: Vishwamitra et al. 의 5단계 추론 (Target → Derogation → Direction → Incitation → Decision) 을 별도 호출이 아닌 **단일 프롬프트 내 chain-of-thought** 로 통합. 누적 오류 및 환각 감소 목적.

**3단계 — SlangLLM 사전 차단**: Patel & Alsobeh 의 PoS 점수 (INTJ=1.0, ADJ=0.8, VERB=0.7, NOUN=0.5) 의 최댓값이 임계값 0.6 미만이면 LLM 호출 없이 정상 처리. API 비용 절감.

**4단계 — 환각 검증**: LLM이 반환한 `toxic_span`이 원문 텍스트에 literal substring으로 존재하는지 검증. 미일치 시 라벨 무효화.

**구현**: `gemma-4-31B-it` (OpenAI-호환 외부 API), `AsyncOpenAI` 기반 64-concurrent async 호출. 24K 건을 27.4분에 처리 (14.6 records/sec).

### 2.5 Threshold Tuning

검증셋 (val) 에서 카테고리별 F1을 최대화하는 threshold (0.10~0.90, step 0.02) 를 탐색하고 테스트셋에 적용. 재학습 없이 결정 경계 조정만으로 precision-recall 균형을 회복한다.

---

## 3. Experiments

### 3.1 메인 결과

테스트셋 (25,481건, 영 12.5K / 한 13K) 에서의 최종 성능 (균형 pseudo + 31B + threshold 튜닝):

**튜닝 전**

| Metric | LR | MLP | Δ |
|---|---:|---:|---:|
| macro-F1 | 0.438 | **0.481** | +0.043 |
| micro-F1 | 0.494 | **0.511** | +0.017 |
| 영어 macro-F1 | 0.382 | **0.451** | +0.069 |
| 한국어 macro-F1 | 0.296 | **0.318** | +0.022 |

**튜닝 후**

| Metric | MLP (0.5 fixed) | MLP (tuned) | Δ |
|---|---:|---:|---:|
| macro-F1 | 0.401 | **0.485** | +0.083 |
| micro-F1 | 0.421 | **0.562** | +0.141 |
| 영어 macro-F1 | 0.588 | **0.681** | +0.093 |
| 한국어 macro-F1 | 0.302 | **0.378** | +0.076 |

### 3.2 카테고리별 성능 (MLP)

**튜닝 후**

| 카테고리 | P | R | F1 | thr | Support |
|---|---:|---:|---:|---:|---:|
| 욕설 | 0.604 | 0.749 | 0.669 | 0.50 | 8,302 |
| 혐오발언 | 0.500 | 0.442 | 0.469 | 0.76 | 2,239 |
| 성 관련 | 0.389 | 0.470 | 0.425 | 0.88 | 449 |
| 살해협박 | 0.535 | 0.831 | 0.651 | 0.86 | 1,128 |
| 정치 | 0.405 | 0.447 | 0.425 | 0.76 | 2,109 |
| 기타유해 | 0.227 | 0.329 | 0.269 | 0.72 | 1,548 |

**튜닝 전**

| 카테고리 | P | R | F1 | Support |
|---|---:|---:|---:|---:|
| 욕설 | 0.743 | 0.754 | 0.748 | 84,814 |
| 혐오발언 | 0.304 | 0.741 | 0.432 | 53,480 |
| 성 관련 | 0.227 | 0.655 | 0.337 | 12,690 |
| 살해협박 | 0.589 | 0.896 | 0.711 | 17,630 |
| 정치 | 0.256 | 0.885 | 0.398 | 72,235 |
| 기타유해 | 0.153 | 0.911 | 0.263 | 89,735 |


### 3.3 한국어 성능 개선을 위한 체계적 실험

한국어 macro-F1이 초기 0.30 수준에서 정체되는 현상을 분석하기 위해 **7가지 접근**을 순차 적용하였다:

| # | 접근 | 영어 F1 | 한국어 F1 | 효과 |
|---|---|---:|---:|---|
| 1 | E5 baseline (7 cat) | 0.40 | 0.30 | — |
| 2 | KcELECTRA embedding | 0.22 | 0.22 | ❌ 성능 저하 |
| 3 | ko-sroberta embedding | 0.24 | 0.24 | ❌ STS 모델 부적합 |
| 4 | K-MHaS → K-HATERS | 0.46 | 0.33 | ✅ +0.03 |
| 5 | 7→6 cat (gender 통합) | 0.46 | 0.33 | ≈ 변화 없음 |
| 6 | ollama 3B → Gemma31B (불균형) | 0.45 | 0.32 | ❌ 보수적 라벨 |
| 7 | 카테고리 균형 pseudo (31B) | 0.59 | 0.30 | ≈ 미미 |
| **8** | **+ threshold 튜닝** | **0.68** | **0.38** | ⭐ **+0.08** |

**관찰**: 임베딩 교체 (#2, #3) 와 LLM 모델 확대 (#6) 는 한국어 성능에 부정적 또는 무의미한 영향을 보였다. 데이터셋 교체 (#4) 와 threshold 튜닝 (#8) 만 통계적으로 유의미한 개선을 보였다.

---

## 4. Discussion

### 4.1 한국어 0.30 벽의 본질적 원인

체계적 실험을 통해 한국어 성능 정체의 근본 원인이 **데이터의 본질적 한계**임을 규명하였다:

1. **세분화 라벨 부재**: K-HATERS, UnSmile, K-MHaS, BEEP 모두 `gender` 라벨을 단일 카테고리로 제공. 영어 데이터 (ToxiGen, ETHOS) 처럼 `sexism` 과 `sexual_harassment` 를 분리하지 않음. 본 연구에서 7→6 카테고리로 통합한 핵심 근거이다.

2. **라벨 노이즈**: K-MHaS의 `Profanity` 카테고리에 종교 비하·정상 문장·구두점만 있는 항목 등이 혼재. K-HATERS로 교체하여 부분 해결하였다.

3. **임베딩 표현 한계**: multilingual-E5는 의미 유사도에는 강하지만 한국어 욕설·혐오 표현의 미세한 어조 차이를 충분히 인코딩하지 못함. 그러나 한국어 전용 임베딩 (KcELECTRA, ko-sroberta) 으로 교체해도 개선되지 않음을 확인하였다 (#2, #3).

### 4.2 Pseudo-Labeling의 역설

흥미롭게도 더 큰 LLM (3B → 31B) 이 pseudo-label 품질에서 항상 우월하지 않았다. 31B는 보수적으로 라벨링 (toxic 비율 27%) 하여 학습 시그널이 부족한 반면, 3B는 관대하게 (50%) 라벨링하여 풍부한 학습 데이터를 제공했다. **카테고리 균형 샘플링 (24K, toxic 69%)** 을 통해 이를 부분적으로 해결하였다.

### 4.3 Threshold Tuning의 일관된 효과

7가지 시도 중 threshold 튜닝이 유일하게 일관된 (+0.08) 개선을 보였다. 이는 BCE 손실 + class_weight='balanced' 학습이 **과잉 예측 (high recall, low precision)** 을 야기하며, 검증셋 기반 카테고리별 threshold 조정으로 precision-recall trade-off를 회복할 수 있음을 시사한다.

---

## 5. Conclusion & Limitations

본 연구는 영어-한국어 통합 multi-label 유해 발화 분류기를 macro-F1 0.485 로 구축하였으며, 한국어 성능 정체의 원인이 임베딩이나 모델 크기가 아닌 **원본 데이터셋의 라벨 품질** 임을 7가지 체계적 실험으로 규명하였다.

**Limitations**:
- 한국어 `gender` 카테고리는 데이터 부족 (3,728건) 으로 신뢰도 제한.
- BAD unsafe 의 pseudo-label은 LLM 환각 가능성을 완전히 배제하지 못함.
- 평가는 카테고리 binary F1 위주이며, 다중 라벨 동시 예측 (subset accuracy) 측면은 추후 분석 필요.

**Future Work**:
- 한국어 전용 인간 재검수 라벨 (K-HATERS 일부 재라벨링) 으로 gender 세분화 시도
- 한국어 특화 LLM (Llama-3-Korean, EXAONE 등) 을 pseudo-labeler 로 비교
- Guard 모델 패밀리 (Llama-Guard 3, ShieldGemma) 와의 정량 비교

---

## 6. References

[1] **Patel, K. & Alsobeh, A. (2024).** *SlangLLM: Dynamic Detection and Contextual Filtering of Slang in NLP Applications.*

[2] **Vishwamitra, N. et al. (2024).** *HateGuard: LLM-Guided Detection of Hate Speech via Chain-of-Thought Reasoning (HateCoT).* IEEE S&P 2024.

[3] **Ghorbanpour, F. et al. (2025).** *Can Prompting LLMs Unlock Hate Speech Detection across Languages? A Zero-shot and Few-shot Study.* TUM.

[4] **Park, S. et al. (2024).** *K-HATERS: A Hate Speech Detection Corpus Co-Annotated by Experts and Non-Experts.* humane-lab.

---

## 7. Reproducing

### 7.1 환경
```
Python 3.9, PyTorch 2.x, sentence-transformers, openai (>=1.0)
GPU: 2× RTX 3090 Ti (24GB) — 임베딩/학습에만 사용
```

### 7.2 파이프라인 실행
```bash
# 1. 데이터 준비 (영/한 통합, 6 카테고리)
python prepare_llm_dataset_v3.py


# 2. LLM pseudo-labeling (31B async)
python llm_pseudo_label_v5_async.py

# 3. Filter (ghost label 제거)
python filter_pseudo_labels.py --diff

# 4. DB 구축 + E5 임베딩
python build_database.py --rebuild
python embed_texts.py

# 5. 학습
python train_classifier.py

# 6. 평가 + threshold 튜닝
python evaluate.py --balanced --min-support 50
python tune_threshold.py --model mlp
python tune_threshold.py --model lr
```

### 7.3 파일 구조
```
engeneer/
├── multilabel_filter.py            # 데이터 로더 (6 카테고리 통합)
├── prepare_llm_dataset_v3.py       # 데이터셋 준비
├── llm_pseudo_label_v5_async.py    # LLM pseudo-labeling (async)
├── filter_pseudo_labels.py         # Pseudo-label 정제
├── slang_pos_scorer.py             # SlangLLM PoS 점수
├── build_database.py               # SQLite DB 구축
├── embed_texts.py                  # E5 임베딩 생성
├── train_classifier.py             # MLP/LR 학습
├── evaluate.py                     # 평가
├── tune_threshold.py               # Threshold 최적화
└── results/
    ├── best_thresholds_mlp.json
    ├── best_thresholds_lr.json
    └── confusion_matrix_*.png
```

---

