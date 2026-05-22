# Korean-English Multi-label Toxic Speech Classification

영어/한국어 혼합 텍스트에 대한 7개 카테고리 multi-label 유해 발언 분류기.
LLM pseudo-labeling + multilingual embedding + 두 분류기(LR/MLP) 비교 + Poison Level 정책.

---

## 1. 개요

### 문제 정의
온라인 댓글·메시지에서 유해 발언을 자동 탐지:
- **다국어** (영어 + 한국어) 단일 모델로 처리
- **세분화된 카테고리** (7종 multi-label)
- **위험도 기반 차단 정책** (BLOCK / FILTER / WARN / PASS)

### 7개 카테고리

| 영문 | 한국어 | 가중치 | 정의 |
|---|---|---:|---|
| `profanity` | 욕설 | 0.7 | 비속어, 욕설 (fuck, 씨발 등) |
| `hate_speech` | 혐오발언 | 0.9 | 인종/종교/성정체성/장애 기반 공격 |
| `sexual_harassment` | 성희롱 | 0.9 | 명시적 성적 발언, 성적 모욕 |
| `sexism` | 성차별 | 0.7 | 성별 기반 차별 |
| `threat` | 살해협박 | **1.0** | 폭력·살해·위해 위협 |
| `political` | 정치 | 0.5 | 정치인/정당/이념 비방 |
| `other` | 기타유해 | 0.4 | 외모비하, 연령차별, 지역차별 등 |

---

## 2. 데이터셋

### 출처 (10개 통합)
**영어**: HateXplain, Ethos, Toxigen, Jigsaw, BAD(safe/unsafe), Hate Speech
**한국어**: K-MHaS, Korean Unsmile, AIHub 대화

### 전처리 정책
- 영어 : 한국어 = **1 : 1**
- 같은 언어 내에서 유해 : 정상 = **1.3 : 1**

---

## 3. 전체 파이프라인 (8단계 + 1)

```
┌─────────────────────────────────────────────────────┐
│         [원본 10개 데이터셋]                          │
└─────────────────┬───────────────────────────────────┘
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [1] prepare_llm_dataset_v3.py                   │
  │     데이터 통합 + 영/한 1:1 균형 + 유해/정상 1.3:1  │
  └─────────────────┬───────────────────────────────┘
                  ↓
        labeled.json + pseudo_target.json
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [2] llm_pseudo_label_v4.py                      │
  │     Ollama Llama 3.2 3B로 의사 라벨링            │
  │     ├ HateCoT in-prompt CoT                     │
  │     └ slang_pos_scorer.py로 사전 차단            │
  └─────────────────┬───────────────────────────────┘
                  ↓
        pseudo_labeled.jsonl.gz (5,000건)
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [3] inspect_pseudo_labels.py  (선택)             │
  │     LLM 라벨링 결과 샘플 검토                     │
  └─────────────────────────────────────────────────┘
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [4] filter_pseudo_labels.py                     │
  │     룰 기반 노이즈 정제                          │
  │     R1: 키워드 검증 / R1_Rescued: 신뢰도 1위 복구  │
  │     R2/R3: 다중라벨/짧은텍스트 제한              │
  └─────────────────┬───────────────────────────────┘
                  ↓
       pseudo_labeled_filtered.jsonl.gz (4,063건)
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [5] build_database.py                           │
  │     SQLite DB 구축 (4개 테이블)                  │
  │     texts / labels / embeddings / splits        │
  └─────────────────┬───────────────────────────────┘
                  ↓
              dataset.sqlite
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [6] embed_texts.py                              │
  │     multilingual-E5-small (384-dim)             │
  └─────────────────┬───────────────────────────────┘
                  ↓
        DB의 embeddings 테이블 (BLOB)
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [7] train_classifier.py                         │
  │     LR (OneVsRest) + MLP 학습                   │
  └─────────────────┬───────────────────────────────┘
                  ↓
       models/lr_model.pkl + mlp_model.pt
                  ↓
  ┌─────────────────────────────────────────────────┐
  │ [8] evaluate.py                                 │
  │     F1/P/R, per-language, confusion matrix      │
  └─────────────────┬───────────────────────────────┘
                  ↓
             results/ (metrics, png)

  ┌─────────────────────────────────────────────────┐
  │ [9] classify.py                                 │
  │     사용자 인터페이스 + Poison Level 정책        │
  │     ├ 단일 문장 / 인터랙티브 / 파일 모드          │
  │     └ PL 계산 → BLOCK/FILTER/WARN/PASS          │
  └─────────────────────────────────────────────────┘
```

### 파일 매핑

| 단계 | 파일 | 입력 | 출력 |
|---|---|---|---|
| 1 | `prepare_llm_dataset_v3.py` | 10개 raw 데이터셋 | `labeled.json`, `pseudo_target.json` |
| 2 | `llm_pseudo_label_v4.py` | `pseudo_target.json` | `pseudo_labeled.jsonl.gz` |
| 3 | `inspect_pseudo_labels.py` | `pseudo_labeled.jsonl.gz` | (콘솔 출력) |
| 4 | `filter_pseudo_labels.py` | `pseudo_labeled.jsonl.gz` | `pseudo_labeled_filtered.jsonl.gz` |
| 5 | `build_database.py` | 위 두 라벨 파일 | `dataset.sqlite` |
| 6 | `embed_texts.py` | `dataset.sqlite` | DB 내 embeddings 테이블 |
| 7 | `train_classifier.py` | DB | `models/lr_model.pkl`, `mlp_model.pt` |
| 8 | `evaluate.py` | DB + 모델 | `results/metrics.json`, `comparison.csv`, `confusion_matrix_*.png` |
| 9 | `classify.py` | 모델 + 사용자 입력 | Action (BLOCK/FILTER/WARN/PASS) |

### 보조 모듈
- **`slang_pos_scorer.py`** — SlangLLM PoS 점수 계산 (영/한)
  - 한국어 강화: 사전(90+) + 패턴(16개) + 한국어 PoS 가중치 + 전체텍스트 스캔
- **`multilabel_filter.py`** — 데이터셋별 라벨 정규화 (이미 보유)

---

## 4. 핵심 기법 (논문 적용)

### 4.1 HateCoT (HateGuard, Ko et al., 2023)
원 논문: 5단계 다중 호출 reasoning
> Target → Derogation → Direction → Incitation → Decision

구현: **1-stage in-prompt CoT**로 압축
- 3-stage 호출 시 3B 모델에서 누적 오류·환각 발생
- 1-stage 프롬프트 안에 5단계 reasoning을 chain-of-thought 텍스트로 통합

### 4.2 SlangLLM (Patel & Alsobeh, 2025) + 한국어 강화

**원 논문 (영어)**: PoS 점수표
| PoS | Score |
|---|---|
| INTJ (감탄사) | 1.0 |
| ADJ (형용사) | 0.8 |
| VERB | 0.7 |
| PROPN | 0.6 |
| NOUN | 0.5 |

**한국어 강화 (이 프로젝트 기여)**: 4단계 결합
- (0) 전체 텍스트 사전/패턴 스캔 ← Okt 토큰 분리 문제 우회
- (1) 토큰별 사전 매칭 → 1.0 (90+ 키워드)
- (2) 토큰별 정규식 패턴 매칭 → 0.9 (변형 욕설: "씨1발", "ㅆㅂ" 등)
- (3) 한국어 특화 PoS 가중치 → 명사 0.5 → **0.7** (한국어 슬랭은 명사형 多)

사용처:
- `llm_pseudo_label_v4.py` — 최대 점수 < 0.6이면 LLM 호출 없이 정상 처리 (`pos_skip`)
- `classify.py` — PL 공식의 `slang_conf` 신호 계산

### 4.3 환각 검증
`llm_pseudo_label_v4.py`에서 LLM이 출력한 `toxic_span`이 실제 텍스트에 substring으로 존재하는지 확인. 없으면 라벨 자동 무효화 (`hallucinated_span`).

### 4.4 룰 기반 노이즈 필터링 (`filter_pseudo_labels.py`)
- **R1**: 카테고리별 키워드 사전 매칭 — 0개면 라벨 제거
- **R1 Rescued**: 모든 라벨이 R1에서 잘렸으면 신뢰도 1위 복구 (toxic 신호 보존)
- **R2**: 라벨 3개+ → 신뢰도 상위 2개만
- **R3**: 짧은 텍스트(<10자)에 라벨 2개+ → 1개만
- **사전 제거**: 의미있는 문자 <5자 텍스트 학습에서 제외

### 4.5 Poison Level (PL) 정책 — `classify.py`
```
PL = 3·slang_conf + 4·cot_confidence + 3·max_category_weight   (0 ≤ PL ≤ 10)
```

**3개 시그널:**
- `slang_conf` — SlangLLM PoS 점수 평균 (0~1)
- `cot_confidence` — 분류기 출력 max 확률 (0~1)
- `max_category_weight` — 탐지 카테고리 중 최대 위험도

**Action 정책:**
| PL 범위 | Action | 의미 |
|---|---|---|
| PL ≥ 7 | BLOCK | 완전 차단 |
| 4 ≤ PL < 7 | FILTER | 유해부 마스킹 |
| 2 ≤ PL < 4 | WARN | 경고 + 사용자 통과 |
| PL < 2 | PASS | 안전 통과 |

**특수 규칙**: `threat` 카테고리 탐지 시 PL 무관 즉시 BLOCK (안전 우선)

---

## 5. 모델 구성

### 임베딩
- **`intfloat/multilingual-e5-small`** (384-dim, 118M params)
- 입력 prefix: `"query: "`
- L2 normalized

### 분류기 (2종 비교)

**Logistic Regression (메인)**
- `OneVsRestClassifier(LogisticRegression(C=1.0, class_weight="balanced"))`
- 7개 binary classifier (multi-label)
- 학습: ~30초

**MLP (비교군)**
- 구조: 384 → 256 → 128 → 7
- ReLU + Dropout 0.2
- `BCEWithLogitsLoss(pos_weight=balanced)`
- AdamW, lr=1e-3, ~2분 (GPU)

---

## 6. 결과

### Test set (balanced)
- 120,084건 (영 61,679 / 한 58,405)
- pseudo + human 라벨 균등 조합

### 전체 성능 비교

| Metric | LR | MLP | Δ |
|---|---:|---:|---:|
| macro-F1 | 0.424 | **0.481** | +5.7% |
| micro-F1 | 0.581 | **0.639** | +5.8% |
| weighted-F1 | 0.641 | **0.675** | +3.3% |

### 카테고리별 (MLP)

| 카테고리 | P | R | F1 | Support |
|---|---:|---:|---:|---:|
| 욕설 | 0.772 | 0.862 | **0.814** | 56,250 |
| 살해협박 | 0.686 | 0.899 | **0.779** | 11,595 |
| 성희롱 | 0.478 | 0.857 | 0.613 | 29,121 |
| 혐오발언 | 0.356 | 0.767 | 0.487 | 19,861 |
| 성차별 | 0.259 | 0.842 | 0.396 | 12,323 |
| 정치 | 0.118 | 0.685 | 0.201 | 89 |
| 기타유해 | 0.066 | 0.092 | 0.077 | 120 |

### 언어별 성능 (MLP)

| Language | n | macro-F1 | micro-F1 |
|---|---:|---:|---:|
| English | 61,679 | 0.490 | 0.760 |
| Korean | 58,405 | 0.295 | 0.512 |

---

## 7. 분석 및 한계

### 잘 작동하는 카테고리 (F1 > 0.6)
- **욕설** (0.81): 명시적 욕설 단어가 강한 신호
- **살해협박** (0.78): "kill", "shoot", "죽이" 동사 키워드 명확
- **성희롱** (0.61): 성적 어휘로 패턴화

### 어려운 카테고리 (F1 < 0.4)
- **성차별** (0.40): 맥락 의존성 큼
- **정치** (0.20): 한국어 특화 어휘, 학습 데이터 부족
- **기타유해** (0.08): 카테고리 정의가 너무 광범위

### 영어 vs 한국어 격차 (0.49 vs 0.30)
원인 3가지:
1. LLM pseudo-label의 한국어 정확도 한계 (Llama 3.2 3B)
2. 한국어 데이터셋 다양성 부족 (영어 6개 vs 한국어 3개)
3. 한국어 표기 다양성 (변형 욕설, 신조어, 형태소)

한국어 강화 SlangLLM은 사전 필터 정확도를 높였지만, 분류기 자체의 한국어 임베딩 품질은 multilingual-E5의 한계로 격차 잔존.

---

## 8. 평가 기준 매핑

### Correctness (40%)
- ✅ Raw data 수집 (10개 데이터셋, 215K건)
- ✅ Database 저장 (SQLite, 4개 테이블)
- ✅ Annotation (LLM pseudo-labeling + 룰 필터링)
- ✅ 분석 모델 통합 (임베딩 + LR + MLP + 평가 + PL 정책)

### Effort (20%)
- 215,086건 통합 데이터셋
- 영/한 동시 처리 (multilingual)
- 7개 multi-label 카테고리
- 3개 논문 기법 적용 + SlangLLM 한국어 강화

### Meaningfulness (20%)
- LR vs MLP 비교 (E5 임베딩 품질 분석)
- 영/한 격차 원인 분석
- 카테고리별 학습 난이도 차이
- 4단계 Poison Level 정책 (BLOCK/FILTER/WARN/PASS)

### Presentation (20%)
- 카테고리별 P/R/F1 표
- Confusion matrix 시각화
- 본 README + 발표 자료
- `classify.py` 라이브 데모

---

## 9. 사용 방법

### 설치
```bash
pip install sentence-transformers torch numpy scikit-learn \
            matplotlib tqdm spacy konlpy requests
python -m spacy download en_core_web_sm
sudo apt install default-jdk fonts-nanum
```

### Ollama 설정
```bash
ollama pull llama3.2:3b
ollama serve &
```

### 전체 파이프라인 실행
```bash
# [1] 데이터 준비
python prepare_llm_dataset_v3.py

# [2] LLM 의사 라벨링 (~1시간)
python llm_pseudo_label_v4.py

# [3] 결과 점검 (선택)
python inspect_pseudo_labels.py --sample 30

# [4] 노이즈 필터링
python filter_pseudo_labels.py --diff

# [5] DB 구축
python build_database.py --rebuild

# [6] E5 임베딩 (~15분, GPU)
python embed_texts.py

# [7] 분류기 학습
python train_classifier.py

# [8] 평가
python evaluate.py --balanced

# [9] 사용자 데모 (PL 정책)
python classify.py "씨발 저 틀딱들"     # 단일 문장
python classify.py                       # 인터랙티브
python classify.py --file inputs.txt    # 파일 일괄
```

### 결과물 위치
```
data/llm_dataset/
├── labeled.json                          # human 라벨
├── pseudo_target.json                    # LLM 라벨 대상
├── pseudo_labeled.jsonl.gz               # LLM 라벨링 결과 (raw)
├── pseudo_labeled_filtered.jsonl.gz      # 필터링 후 (최종)
└── dataset.sqlite                        # 통합 DB

models/
├── lr_model.pkl
└── mlp_model.pt

results/
├── metrics.json
├── comparison.csv
├── confusion_matrix_lr.png
└── confusion_matrix_mlp.png
```

---

## 10. classify.py 사용 예시

```bash
$ python classify.py "씨발 좀 닥쳐"

입력: 씨발 좀 닥쳐
──────────────────────────────────────────────────────
  [BLOCK] BLOCK
  사유: PL 8.72 >= 7.0 (완전 차단)

  Poison Level 분석:
    slang_conf      = 1.000  (x 3.0)
    cot_confidence  = 0.945  (x 4.0)
    max_cat_weight  = 0.700  (x 3.0)
    PL = 8.72 / 10.00   [##########################----]

  탐지된 카테고리:
      욕설        score=0.945  weight=0.7
```

---

## 11. 향후 개선 방향

1. **더 큰 LLM** — Llama 3.3 70B로 pseudo-labeling 재실행 시 노이즈 감소 예상
2. **한국어 특화 임베딩** — KcELECTRA 등으로 한국어 분류기 별도 학습 + 언어별 라우팅
3. **추가 한국어 데이터** — K-HATERS, KOLD 등
4. **카테고리 재정의** — "기타유해" 분할 (외모/연령/지역)
5. **인간 검증 라벨** — 일부 샘플 수동 검수로 정답셋 확보

---

## 12. 참고문헌

1. **HateGuard** (Ko et al., 2023) — *Dynamic Hate Speech Detection through HateCoT*. arXiv:2312.15099
2. **SlangLLM** (Patel & Alsobeh, 2025) — *Dynamic Detection and Contextual Filtering of Slang in NLP Applications*
3. **Plaza-del-Arco et al.** — *Can Prompting LLMs Unlock Hate Speech Detection across Languages?*
4. **multilingual-E5** (Wang et al.) — `intfloat/multilingual-e5-small`

---

## 13. 라이선스 / 출처

- 원본 데이터셋은 각 출처의 라이선스를 따름
- 본 프로젝트 코드는 학술 목적
