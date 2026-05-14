# Korean-English Multi-label Toxic Speech Classification

영어/한국어 혼합 텍스트에 대한 7개 카테고리 multi-label 유해 발언 분류기.
LLM pseudo-labeling + multilingual embedding + 두 분류기(LR/MLP) 비교 평가.

---

## 1. 개요

### 문제 정의
온라인 댓글·메시지에서 유해 발언을 자동 탐지하되,
- **다국어** (영어 + 한국어)를 하나의 모델로 처리
- **세분화된 카테고리** (단순 toxic/clean이 아닌 7종 분류)
- **다중 라벨** (한 문장이 동시에 여러 카테고리에 속할 수 있음)

### 7개 카테고리

| 카테고리 (영) | 한국어 표기 | 정의 |
|---|---|---|
| `profanity` | 욕설 | 비속어, 욕설 (fuck, 씨발 등) |
| `hate_speech` | 혐오발언 | 인종/종교/성정체성/장애 기반 공격 |
| `sexual_harassment` | 성희롱 | 명시적 성적 발언, 성적 모욕 |
| `sexism` | 성차별 | 성별 기반 차별 발언 |
| `threat` | 살해협박 | 폭력·살해·위해 위협 |
| `political` | 정치 | 정치인/정당/이념 비방 (attack + hate 통합) |
| `other` | 기타유해 | 외모비하, 연령차별, 지역차별 등 |

---

## 2. 데이터셋

### 출처 (10개 통합)

**영어:**
- HateXplain, Ethos, Toxigen, Jigsaw, BAD (safe + unsafe), Hate Speech

**한국어:**
- K-MHaS, Korean Unsmile, AIHub 대화

### 전처리
1. 데이터셋별로 카테고리 매핑 (`multilabel_filter.py`)
2. 영어 : 한국어 = **1 : 1** 비율 유지
3. 같은 언어 내에서 **유해(toxic) : 정상 = 1.3 : 1** 균형
4. `labeled.json` (인간 라벨, ~215K건) + `pseudo_target.json` (LLM 라벨 대상, 5K건)

---

## 3. 파이프라인

```
[원본 10개 데이터셋]
        ↓
   prepare_llm_dataset_v3.py
        ↓
[labeled.json + pseudo_target.json]
        ↓
   llm_pseudo_label_v4.py  ← HateCoT + SlangLLM 적용
        ↓
[pseudo_labeled.jsonl.gz (5K건)]
        ↓
   filter_pseudo_labels.py  ← 룰 기반 노이즈 정제
        ↓
[pseudo_labeled_filtered.jsonl.gz (4K건)]
        ↓
   build_database.py        ← SQLite DB 구축
        ↓
[dataset.sqlite]
        ↓
   embed_texts.py           ← multilingual-E5-small 임베딩
        ↓
[384-dim vectors in DB]
        ↓
   train_classifier.py      ← LR (OvR) + MLP 학습
        ↓
[lr_model.pkl, mlp_model.pt]
        ↓
   evaluate.py              ← F1/P/R, per-language, confusion matrix
        ↓
[results/]
```

### 단계별 파일

| 단계 | 스크립트 | 출력 |
|---|---|---|
| 1. 데이터 준비 | `prepare_llm_dataset_v3.py` | `labeled.json`, `pseudo_target.json` |
| 2. LLM 의사 라벨링 | `llm_pseudo_label_v4.py` | `pseudo_labeled.jsonl.gz` |
| 3. 노이즈 필터링 | `filter_pseudo_labels.py` | `pseudo_labeled_filtered.jsonl.gz` |
| 4. DB 구축 | `build_database.py` | `dataset.sqlite` |
| 5. 임베딩 | `embed_texts.py` | DB의 embeddings 테이블 |
| 6. 분류기 학습 | `train_classifier.py` | `models/lr_model.pkl`, `models/mlp_model.pt` |
| 7. 평가 | `evaluate.py` | `results/metrics.json`, `comparison.csv`, `confusion_matrix_*.png` |

### 보조 파일
- `slang_pos_scorer.py` — SlangLLM PoS 점수 계산 (영/한)
- `inspect_pseudo_labels.py` — 라벨링 결과 점검
- `multilabel_filter.py` — 데이터셋별 라벨 정규화 (이미 보유)

---

## 4. 핵심 기법 (논문 적용)

### 4.1 HateCoT (HateGuard, Ko et al., 2023)
원 논문은 5단계 다중 호출 reasoning을 제안:
> Target → Derogation → Direction → Incitation → Decision

구현에서 시행착오 끝에 **1-stage in-prompt CoT**로 압축:
- 3-stage 다중 호출 시 LLM이 누적 오류·환각을 일으킴 (3B 모델 한계)
- 1-stage 프롬프트 안에 5단계 reasoning을 chain-of-thought 텍스트로 통합

### 4.2 SlangLLM (Patel & Alsobeh, 2025)
PoS 점수표 기반 사전 필터링:

| PoS | Score | 의미 |
|---|---|---|
| INTJ (감탄사) | 1.0 | "fire", "lit", "ㅋㅋ" |
| ADJ (형용사) | 0.8 | "sus", "dope" |
| VERB | 0.7 | "yeet", "kill" |
| PROPN (고유) | 0.6 | 인명 |
| NOUN | 0.5 | 일반 명사 |
| 기타 | 0.2 | — |

**최대 PoS 점수 < 0.6** → LLM 호출 없이 정상 처리 (속도 향상)

### 4.3 환각 검증
LLM이 출력한 `toxic_span`이 실제 텍스트에 substring으로 존재하는지 확인.
없으면 라벨 자동 무효화.

### 4.4 룰 기반 노이즈 필터링 (`filter_pseudo_labels.py`)
LLM 라벨 노이즈를 키워드 사전으로 정제:
- **R1**: 카테고리별 키워드 매칭 — 키워드 0개면 라벨 제거
- **R1 Rescued**: 모든 라벨이 잘렸을 때 신뢰도 1위 1개 복구
- **R2**: 라벨 3개+ → 신뢰도 상위 2개만
- **R3**: 짧은 텍스트(<10자)에 라벨 2개+ → 1개만
- **사전 제거**: 의미있는 문자 <5자 텍스트는 학습에서 제외

---

## 5. 모델 구성

### 임베딩
- **`intfloat/multilingual-e5-small`** (384-dim, 118M params)
- 입력 prefix: `"query: "`
- L2 normalized (cosine similarity 친화적)

### 분류기 (2종 비교)

**Logistic Regression (메인)**
- `OneVsRestClassifier(LogisticRegression(C=1.0, class_weight="balanced"))`
- 7개 binary classifier (multi-label)
- 학습 시간: ~30초

**MLP (비교군)**
- 구조: 384 → 256 → 128 → 7
- Activation: ReLU, Dropout 0.2
- Loss: `BCEWithLogitsLoss(pos_weight=balanced)`
- Optimizer: AdamW, lr=1e-3
- 학습 시간: ~2분 (GPU)

---

## 6. 결과

### 평가 환경
- **Test set (balanced)**: 120,084건 (영 61,679 / 한 58,405)
- pseudo + human 라벨 균등 조합
- min_support 50 (모든 카테고리 평가 가능)

### 전체 성능 비교

| Metric | LR | MLP | Δ |
|---|---:|---:|---:|
| macro-F1 | 0.424 | **0.481** | +5.7% |
| micro-F1 | 0.581 | **0.639** | +5.8% |
| weighted-F1 | 0.641 | **0.675** | +3.3% |

### 카테고리별 (MLP 기준)

| 카테고리 | Precision | Recall | F1 | Support |
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
- **살해협박** (0.78): "kill", "shoot", "죽이" 같은 동사 키워드 명확
- **성희롱** (0.61): 성적 어휘로 패턴화

### 어려운 카테고리 (F1 < 0.4)
- **성차별** (0.40): 맥락 의존성 큼. 단순 성별 단어로 판단 어려움
- **정치** (0.20): 한국어 특화 어휘 (정치인 이름, 신조어). 학습 데이터 부족
- **기타유해** (0.08): 카테고리 정의가 너무 광범위 (외모/연령/지역 등 잡종)

### 영어 vs 한국어 격차 (0.49 vs 0.30)

격차 원인 3가지:

1. **Pseudo-label LLM(Llama 3.2 3B)의 한국어 정확도 한계**
   - 영어 학습 데이터 비중이 압도적
   - 한국어 신조어/변형 욕설 ("ㅆ1발", "틀딱") 이해도 낮음

2. **데이터셋 다양성 불균형**
   - 영어: 6개 출처 (Jigsaw, Ethos, HateXplain 등)
   - 한국어: 3개 출처 (K-MHaS, Unsmile, AIHub)

3. **한국어의 표기 다양성**
   - 변형 욕설 회피 (씨발 → 씨1발, ㅆㅂ)
   - 형태소 분석 어려움
   - Okt 태거가 "씨발"을 명사로 분류 등

### Pseudo-labeling 신뢰도

LLM 라벨링 + 룰 필터링 후에도 약 50.8%가 toxic으로 분류됐는데,
**일부 false positive 잔존** (LLM이 정상 텍스트에 카테고리 라벨 박는 패턴):
- "I'm so angry" → 살해협박 (rescue rule로 1개 라벨 잔존)
- "win medals" → 성차별

이는 3B 작은 모델의 한계로 추정. 더 큰 모델(70B+) 사용 시 개선 여지.

### LR vs MLP

MLP가 일관되게 약 5% 우세하나, **차이가 크지 않음**.
→ E5 임베딩이 의미 공간을 잘 정리해놨기 때문에 선형 분류기로도 충분.

---

## 8. 평가 기준 매핑

### Correctness (40%)
- ✅ Raw data 수집 (10개 데이터셋, 215K건)
- ✅ Database 저장 (SQLite, 4개 테이블)
- ✅ Annotation (LLM pseudo-labeling + 룰 필터링)
- ✅ 분석 모델 통합 (임베딩 + LR + MLP + 평가)

### Effort (20%)
- 215,086건 통합 데이터셋
- 영/한 동시 처리 (multilingual 모델)
- 7개 multi-label 카테고리
- 3개 논문 기법 적용 (HateGuard, SlangLLM, multilingual E5)

### Meaningfulness (20%)
- LR vs MLP 비교 분석
- 영/한 언어별 성능 격차 원인 분석
- 카테고리별 학습 난이도 차이 분석
- LLM pseudo-labeling 한계 인정 및 룰 필터링 보완

### Presentation (20%)
- 카테고리별 P/R/F1 표
- macro/micro/weighted F1 다각도 평가
- Confusion matrix 시각화
- 언어별 성능 비교
- 본 README + 발표 자료

---

## 9. 사용 방법

### 설치
```bash
pip install sentence-transformers torch numpy scikit-learn \
            matplotlib tqdm spacy konlpy requests
python -m spacy download en_core_web_sm
sudo apt install default-jdk fonts-nanum  # KoNLPy + 한글 폰트
```

### Ollama 설정 (LLM pseudo-labeling용)
```bash
# Ollama 설치 후
ollama pull llama3.2:3b
ollama serve  # 백그라운드 실행
```

### 실행 (전체 파이프라인)
```bash
# 1) 데이터셋 준비
python prepare_llm_dataset_v3.py

# 2) LLM pseudo-labeling (5,000건, ~1시간)
python llm_pseudo_label_v4.py

# 3) 노이즈 필터링
python filter_pseudo_labels.py

# 4) DB 구축
python build_database.py --rebuild

# 5) 임베딩 (~15분, GPU 사용 시)
python embed_texts.py

# 6) 분류기 학습
python train_classifier.py

# 7) 평가
python evaluate.py --balanced
```

### 결과물
```
data/llm_dataset/dataset.sqlite          # 통합 DB
models/lr_model.pkl                       # LR 모델
models/mlp_model.pt                       # MLP 모델
results/metrics.json                      # 메트릭 (JSON)
results/comparison.csv                    # 카테고리별 비교
results/confusion_matrix_lr.png           # LR Confusion Matrix
results/confusion_matrix_mlp.png          # MLP Confusion Matrix
```

---

## 10. 향후 개선 방향

1. **더 큰 LLM 모델** — Llama 3.3 70B로 pseudo-labeling 재실행 시 노이즈 큰 폭 감소 예상
2. **한국어 강화** — kogpt, koalpaca 등 한국어 특화 LLM으로 ko 라벨 보강
3. **카테고리 재정의** — "기타유해"가 너무 광범위. 외모/연령/지역으로 분할 검토
4. **인간 검증 라벨** — 일부 샘플 수동 검수로 정답셋 확보 → 평가 신뢰도 ↑
5. **임베딩 fine-tuning** — E5를 toxic detection 도메인에 맞춰 추가 학습

---

## 11. 참고문헌

1. **HateGuard** (Ko et al., 2023) — *Dynamic Hate Speech Detection through HateCoT*. arXiv:2312.15099
2. **SlangLLM** (Patel & Alsobeh, 2025) — *Dynamic Detection and Contextual Filtering of Slang in NLP Applications*
3. **Plaza-del-Arco et al.** — *Can Prompting LLMs Unlock Hate Speech Detection across Languages?*
4. **multilingual-E5** (Wang et al.) — `intfloat/multilingual-e5-small`

---

## 12. 라이선스 / 출처

- 원본 데이터셋은 각 출처의 라이선스를 따름
- 본 프로젝트 코드는 학술 목적
