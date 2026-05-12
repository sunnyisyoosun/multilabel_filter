"""
multilabel_filter.py
=======================
2단계 학습 파이프라인:
    Stage 1 - jigsaw + hate_speech + toxigen 으로 1차 모델 학습
    Stage 2 - 1차 모델로 BAD unsafe 재레이블링 후 전체 데이터로 최종 모델 학습

카테고리 (다중 레이블):
    profanity         - 욕설
    hate_speech       - 인종차별
    sexual_harassment - 성희롱
    threat            - 살해 협박

실행:
    python multilabel_filter.py --train
    python multilabel_filter.py --filter "You racist idiot, I'll kill you"
    python multilabel_filter.py --interactive
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# TensorFlow는 CPU만 사용 (GPU는 PyTorch E5 임베딩 전용)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
import json, hashlib, datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = ["profanity", "hate_speech", "sexual_harassment", "sexism", "threat"]
CATEGORY_KO = {
    "profanity":         "욕설",
    "hate_speech":       "인종차별",
    "sexual_harassment": "성희롱",
    "sexism":            "성차별",
    "threat":            "살해 협박",
}
MODEL_PATH  = "multilabel_filter.pkl"
RELABEL_THR = 0.50
# 카테고리별 재레이블링 임계값
# threat: Stage 1 F1=0.92로 충분히 신뢰할 수 있으므로 재활성화
# 단 threshold를 높게(0.60) 잡아 확실한 것만 재레이블
RELABEL_THR_PER_CAT = {
    "profanity":         0.55,
    "hate_speech":       0.55,
    "sexual_harassment": 0.50,
    "sexism":            0.50,
    "threat":            0.75,   # E5 모델이 threat 과도분류 → threshold 높임
}
FILTER_THR  = 0.40
# 카테고리별 필터링 threshold (오탐 방지)
FILTER_THR_PER_CAT = {
    "profanity":         0.40,
    "hate_speech":       0.45,
    "sexual_harassment": 0.85,  # 오탐 많아서 높임
    "sexism":            0.50,
    "threat":            0.45,
}

# ── 데이터 엔지니어링 경로 ──
DATA_DIR       = Path("data")
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EMB_DIR        = DATA_DIR / "embeddings"
META_DIR       = DATA_DIR / "metadata"
for _d in [RAW_DIR, PROCESSED_DIR, EMB_DIR, META_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_bad() -> tuple:
    """
    BAD → safe / unsafe 분리.
    HuggingFace datasets 로 로드 (TFDS 대신) → TensorFlow GPU 점유 문제 해결.
    로컬 캐시 없으면 HF에서 다운로드.
    """
    logger.info("BAD 로드 중 (HuggingFace)...")
    try:
        from datasets import load_dataset
        safe_rows, unsafe_rows = [], []
        for split in ["train", "validation", "test"]:
            try:
                ds = load_dataset(
                    "facebook/bot_adversarial_dialogues",
                    split=split,
                )
            except Exception:
                # 미러 시도
                ds = load_dataset(
                    "allenai/bot_adversarial_dialogue",
                    split=split,
                )
            df = ds.to_pandas()
            # 컬럼명 통일
            if "dialogue" in df.columns and "text" not in df.columns:
                df["text"] = df["dialogue"]
            df["text"] = df["text"].astype(str)
            if "labels" not in df.columns and "label" in df.columns:
                df["labels"] = df["label"]
            df["labels"] = df["labels"].astype(int)
            safe_rows.append(df[df["labels"] == 0][["text"]])
            unsafe_rows.append(df[df["labels"] == 1][["text"]])

        safe_df   = pd.concat(safe_rows,   ignore_index=True)
        unsafe_df = pd.concat(unsafe_rows, ignore_index=True)
        for cat in CATEGORIES:
            safe_df[cat] = 0
        logger.info(f"  BAD safe  : {len(safe_df):,}건")
        logger.info(f"  BAD unsafe: {len(unsafe_df):,}건 (재레이블링 예정)")
        return safe_df, unsafe_df

    except Exception as e:
        logger.warning(f"  HuggingFace BAD 로드 실패 ({e}), TFDS 로컬 캐시로 시도...")
        # TFDS 폴백 — GPU 메모리 선점 최소화
        import os, tensorflow_datasets as tfds
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # TFDS 로드 중 GPU 숨기기
        safe_rows, unsafe_rows = [], []
        for split in ["train", "valid", "test"]:
            ds, ds_info = tfds.load(
                "bot_adversarial_dialogue/dialogue_datasets",
                split=split, with_info=True, shuffle_files=False,
            )
            df = tfds.as_dataframe(ds, ds_info)
            df["text"] = df["text"].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else str(v)
            )
            df["labels"] = df["labels"].astype(int)
            safe_rows.append(df[df["labels"] == 0][["text"]])
            unsafe_rows.append(df[df["labels"] == 1][["text"]])
        # TFDS 끝나면 GPU 다시 복원
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        safe_df   = pd.concat(safe_rows,   ignore_index=True)
        unsafe_df = pd.concat(unsafe_rows, ignore_index=True)
        for cat in CATEGORIES:
            safe_df[cat] = 0
        logger.info(f"  BAD safe  : {len(safe_df):,}건")
        logger.info(f"  BAD unsafe: {len(unsafe_df):,}건 (재레이블링 예정)")
        return safe_df, unsafe_df


def load_jigsaw() -> pd.DataFrame:
    """
    jigsaw_toxicity — tasksource 미러 사용 (google/jigsaw_toxicity_pred는 스크립트 전용).
    tasksource/jigsaw_toxicity: Parquet 변환본, 동일 컬럼 구조.
    toxic → profanity / identity_hate → hate_speech
    obscene → sexual_harassment / threat → threat
    """
    logger.info("jigsaw_toxicity 로드 중...")
    try:
        from datasets import load_dataset
        # tasksource 미러: 스크립트 없이 Parquet으로 바로 로드 가능
        ds = load_dataset("tasksource/jigsaw_toxicity", split="train")
        df = ds.to_pandas()
        logger.info(f"  jigsaw 컬럼: {list(df.columns)}")

        text_col = next((c for c in ["comment_text", "text", "comment"] if c in df.columns), None)
        if text_col is None:
            raise KeyError(f"텍스트 컬럼 없음. 실제: {list(df.columns)}")
        df["text"]              = df[text_col].astype(str)
        df["profanity"]         = df.get("toxic",         pd.Series(0, index=df.index)).astype(int)
        df["hate_speech"]       = df.get("identity_hate", pd.Series(0, index=df.index)).astype(int)
        df["sexual_harassment"] = df.get("obscene",       pd.Series(0, index=df.index)).astype(int)
        df["sexism"]            = 0
        df["threat"]            = df.get("threat",        pd.Series(0, index=df.index)).astype(int)

        # threat==1 행만 추출 — Jigsaw는 threat 카테고리 전용 소스
        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        result = result[result["threat"] == 1].reset_index(drop=True)
        logger.info(f"  jigsaw: {len(result):,}건  (threat 전용, threat==1 필터링)")
        return result
    except Exception as e:
        logger.warning(f"  jigsaw 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_hate_speech() -> pd.DataFrame:
    logger.info("hate_speech_offensive 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tdavidson/hate_speech_offensive", split="train")
        df = ds.to_pandas().rename(columns={"tweet": "text"})
        df["text"]              = df["text"].astype(str)
        df["profanity"]         = (df["class"] == 1).astype(int)
        df["hate_speech"]       = (df["class"] == 0).astype(int)
        df["sexual_harassment"] = 0
        df["sexism"]            = 0
        df["threat"]            = 0
        result = df[["text"] + CATEGORIES]
        logger.info(f"  hate_speech_offensive: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  hate_speech_offensive 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_toxigen() -> pd.DataFrame:
    """
    toxigen-data: 실제 텍스트 컬럼은 'generation' (text 아님)
    toxigen/toxigen-data 로 namespace 수정
    """
    logger.info("toxigen 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("toxigen/toxigen-data", split="train")
        df = ds.to_pandas()
        logger.info(f"  toxigen 실제 컬럼: {list(df.columns)}")

        # 텍스트 컬럼 탐색
        for col in ["generation", "text", "prompt"]:
            if col in df.columns:
                df["text"] = df[col].astype(str)
                break
        else:
            raise KeyError(f"텍스트 컬럼 없음. 실제 컬럼: {list(df.columns)}")

        # toxicity 컬럼 탐색
        tox_col = next(
            (c for c in ["toxicity_human", "prompt_label", "label"] if c in df.columns),
            None
        )
        if tox_col:
            df["is_toxic"] = (pd.to_numeric(df[tox_col], errors="coerce").fillna(0) >= 0.5).astype(int)
        else:
            df["is_toxic"] = 0

        RACE_GROUPS   = {"asian", "black", "latino", "native_american", "middle_east", "jewish"}
        GENDER_GROUPS  = {"lgbtq", "trans"}
        SEXISM_GROUPS  = {"women"}

        def map_labels(row):
            grp = str(row.get("target_group", "")).lower()
            tox = row["is_toxic"]
            return {
                "profanity":         0,
                "hate_speech":       int(tox and any(g in grp for g in RACE_GROUPS)),
                "sexual_harassment": int(tox and any(g in grp for g in GENDER_GROUPS)),
                "sexism":            int(tox and any(g in grp for g in SEXISM_GROUPS)),
                "threat":            0,
            }

        mapped = df.apply(map_labels, axis=1, result_type="expand")
        result = pd.concat([df[["text"]], mapped], axis=1)[["text"] + CATEGORIES]
        logger.info(f"  toxigen: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  toxigen 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_hatexplain() -> pd.DataFrame:
    """
    HateXplain — GitHub raw JSON 직접 로드.
    HuggingFace는 스크립트 전용이라 Parquet 변환본 없음.
    원본 데이터: hate-alert/HateXplain GitHub repo
    hate=0 → profanity + hate_speech
    offensive=2 → profanity
    violence rationale → threat
    """
    logger.info("hatexplain 로드 중...")
    try:
        import requests
        from collections import Counter

        url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw = resp.json()

        rows = []
        for post_id, item in raw.items():
            try:
                # majority label 추출 (0=hate, 1=normal, 2=offensive)
                labels = [a["label"] for a in item.get("annotators", [])]
                if not labels:
                    continue
                majority = Counter(labels).most_common(1)[0][0]

                # 토큰 → 문자열
                tokens = item.get("post_tokens", [])
                text = " ".join(tokens) if tokens else ""
                if len(text.strip()) < 5:
                    continue

                profanity         = int(majority in [0, 2])
                hate_speech       = int(majority == 0)
                sexual_harassment = 0
                sexism            = 0

                # rationale에 violence 포함 여부
                rationales = item.get("rationales", [])
                has_violence = int(any(
                    "violence" in str(r).lower() for r in rationales
                ))

                rows.append({
                    "text": text,
                    "profanity": profanity,
                    "hate_speech": hate_speech,
                    "sexual_harassment": sexual_harassment,
                    "sexism": sexism,
                    "threat": has_violence,
                })
            except Exception:
                continue

        df = pd.DataFrame(rows)
        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  hatexplain: {len(result):,}건  (threat 양성: {int(result['threat'].sum())}건)")
        return result
    except Exception as e:
        logger.warning(f"  hatexplain 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_ethos() -> pd.DataFrame:
    """
    ETHOS multilabel — Parquet 직접 로드 (스크립트 방식 지원 종료 대응).
    violence → threat (핵심)
    gender   → sexual_harassment + sexism
    race / national_origin → hate_speech
    """
    logger.info("ETHOS 로드 중...")
    try:
        import requests, io
        url = "https://huggingface.co/datasets/iamollas/ethos/resolve/refs%2Fconvert%2Fparquet/multilabel/train/0000.parquet"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            url2 = "https://huggingface.co/datasets/iamollas/ethos/resolve/main/data/multilabel-train.parquet"
            resp = requests.get(url2, timeout=60)
        df = pd.read_parquet(io.BytesIO(resp.content))
        logger.info(f"  ethos 컬럼: {list(df.columns)}")

        df["text"] = df.get("comment", df.get("text", pd.Series("", index=df.index))).astype(str)

        def to_int_col(name):
            return df.get(name, pd.Series(0, index=df.index)).fillna(0).clip(0, 1).astype(int)

        df["profanity"]         = to_int_col("directed_vs_generalized")
        df["hate_speech"]       = (to_int_col("race") + to_int_col("national_origin")).clip(0, 1)
        gender_col              = to_int_col("gender")
        df["sexual_harassment"] = gender_col
        df["sexism"]            = gender_col
        df["threat"]            = to_int_col("violence")   # ← threat 핵심 소스

        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  ethos: {len(result):,}건  (threat 양성: {int(result['threat'].sum())}건)")
        return result
    except Exception as e:
        logger.warning(f"  ethos 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)
        logger.warning(f"  ethos 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_kmhas() -> pd.DataFrame:
    """
    K-MHaS — load_dataset으로 parquet 변환본 직접 지정.
    스크립트 방식 종료 대응: revision='refs/convert/parquet' 사용.
    0=Politics, 1=Origin, 2=Physical, 3=Age,
    4=Gender, 5=Religion, 6=Race, 7=Profanity, 8=Not Hate Speech
    """
    logger.info("K-MHaS (한국어) 로드 중...")
    try:
        from datasets import load_dataset

        # parquet 변환본 revision으로 스크립트 우회
        ds = load_dataset(
            "jeanlee/kmhas_korean_hate_speech",
            revision="refs/convert/parquet",
            split="train",
        )
        df = ds.to_pandas()
        logger.info(f"  K-MHaS 컬럼: {list(df.columns)}")

        df["text"] = df.get("text", df.get("comment", pd.Series("", index=df.index))).astype(str)

        def extract(row, targets):
            try:
                labels = row["label"] if isinstance(row["label"], list) else [row["label"]]
                return int(any(l in targets for l in labels))
            except Exception:
                return 0

        df["profanity"]         = df.apply(lambda r: extract(r, [7]), axis=1)
        df["hate_speech"]       = df.apply(lambda r: extract(r, [1, 6]), axis=1)
        df["sexual_harassment"] = df.apply(lambda r: extract(r, [2, 3]), axis=1)
        df["sexism"]            = df.apply(lambda r: extract(r, [4]), axis=1)
        df["threat"]            = 0

        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  K-MHaS: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  K-MHaS 로드 최종 실패: {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_korean_unsmile() -> pd.DataFrame:
    """
    Korean UnSmile (Smilegate AI) — 한국어 혐오 표현 멀티레이블.
    악플/욕설 → profanity
    여성/가족, 남성, 성소수자 → sexual_harassment
    인종/국적 → hate_speech
    """
    logger.info("Korean UnSmile 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("smilegate-ai/kor_unsmile", split="train")
        df = ds.to_pandas()
        logger.info(f"  UnSmile 컬럼: {list(df.columns)}")

        df["text"] = df.get("문장", df.get("text", pd.Series("", index=df.index))).astype(str)

        # 컬럼명 한글 그대로 매핑
        def col(name):
            return df.get(name, pd.Series(0, index=df.index)).fillna(0).astype(float)

        df["profanity"]         = (col("악플/욕설") >= 0.5).astype(int)
        df["hate_speech"]       = (col("인종/국적") >= 0.5).astype(int)
        df["sexual_harassment"] = (col("성소수자") >= 0.5).astype(int)           # 성소수자 차별 → 성희롱
        df["sexism"]            = ((col("여성/가족") + col("남성")) >= 0.5).clip(0, 1).astype(int)  # 젠더 기반 차별 → 성차별
        df["threat"]            = 0

        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  Korean UnSmile: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  Korean UnSmile 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

class E5Embedder(BaseEstimator, TransformerMixin):
    """
    intfloat/multilingual-e5-small 임베딩 transformer.
    CUDA 자동 감지, OOM 방지를 위해 batch_size=32 (CUDA) / 64 (CPU).
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._batch_size = 32

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            if torch.cuda.is_available():
                device = "cuda"
                self._batch_size = self.batch_size or 32
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "cpu"
                self._batch_size = self.batch_size or 64
            else:
                device = "cpu"
                self._batch_size = self.batch_size or 64
            logger.info(f"  E5 모델 로드 중: {self.model_name}  (device={device}, batch={self._batch_size})")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit(self, X, y=None):
        self._load_model()
        return self

    def transform(self, X):
        import torch
        model = self._load_model()
        texts = list(X) if not isinstance(X, list) else X
        texts = [f"passage: {t}" for t in texts]

        # fp16으로 메모리 절반 절약, 청크 단위로 인코딩
        CHUNK = 10000
        all_embeddings = []
        for start in range(0, len(texts), CHUNK):
            chunk = texts[start:start + CHUNK]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                emb = model.encode(
                    chunk,
                    batch_size=self._batch_size,
                    show_progress_bar=len(texts) > 1000,
                    normalize_embeddings=True,

                )
            all_embeddings.append(emb)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        import numpy as np
        return np.vstack(all_embeddings).astype("float32")


def build_pipeline() -> Pipeline:
    """
    multilingual-e5-small 임베딩 + MultiOutputClassifier(LogisticRegression)
    TF-IDF+RF 대비 한국어/영어 혼합 데이터에서 성능 향상 기대.
    """
    return Pipeline([
        ("emb", E5Embedder()),
        ("clf", MultiOutputClassifier(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
                n_jobs=-1,
            )
        )),
    ])


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"

def load_or_cache(name: str, loader_fn) -> pd.DataFrame:
    """
    Parquet 캐시가 있으면 로드, 없으면 loader_fn() 실행 후 저장.
    재실행 시 HuggingFace 다운로드 불필요.
    """
    path = _cache_path(name)
    if path.exists():
        logger.info(f"  [{name}] 캐시 로드: {path}")
        return pd.read_parquet(path)
    df = loader_fn()
    if len(df) > 0:
        df.to_parquet(path, index=False)
        logger.info(f"  [{name}] Parquet 저장: {path}  ({len(df):,}건)")
    return df


# ──────────────────────────────────────────────
# 2. 임베딩 캐싱 유틸
# ──────────────────────────────────────────────

def _emb_cache_path(texts_hash: str, stage: str) -> Path:
    return EMB_DIR / f"emb_{stage}_{texts_hash}.npy"

def _hash_texts(texts) -> str:
    """텍스트 리스트의 해시값 — 데이터 변경 감지용"""
    h = hashlib.md5("|".join(texts[:100]).encode()).hexdigest()[:8]
    return f"{len(texts)}_{h}"

def embed_with_cache(embedder, texts: list, stage: str) -> np.ndarray:
    """
    임베딩 캐시가 있으면 로드, 없으면 계산 후 npy로 저장.
    226,680건 재임베딩 방지 (50초 → 0초).
    """
    h = _hash_texts(texts)
    cache = _emb_cache_path(h, stage)
    if cache.exists():
        logger.info(f"  [{stage}] 임베딩 캐시 로드: {cache}")
        return np.load(cache)
    logger.info(f"  [{stage}] 임베딩 계산 중... ({len(texts):,}건)")
    emb = embedder.transform(texts)
    np.save(cache, emb)
    logger.info(f"  [{stage}] 임베딩 저장: {cache}")
    return emb


# ──────────────────────────────────────────────
# 3. 메타데이터 저장
# ──────────────────────────────────────────────

def save_metadata(run_info: dict) -> None:
    """학습 실행 정보를 JSON으로 저장 — 버전 관리 및 재현성"""
    run_id  = run_info.get("run_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    path    = META_DIR / f"run_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    # latest 심볼 역할 — 항상 최신 run 덮어쓰기
    latest = META_DIR / "latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    logger.info(f"  메타데이터 저장: {path}")


# ──────────────────────────────────────────────
# 4. 데이터 품질 검증
# ──────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame, label: str) -> None:
    """
    학습 전 데이터 품질 검증.
    실패 시 경고만 출력 (학습은 계속 진행).
    """
    issues = []

    # 결측치 체크
    null_cnt = df["text"].isnull().sum()
    if null_cnt > 0:
        issues.append(f"text 결측치 {null_cnt}건")

    # 길이 이상치 (3자 이하)
    short_cnt = (df["text"].str.len() <= 3).sum()
    if short_cnt > 0:
        issues.append(f"길이 3↓ 텍스트 {short_cnt}건")

    # 중복 텍스트
    dup_cnt = df["text"].duplicated().sum()
    if dup_cnt > 0:
        issues.append(f"중복 텍스트 {dup_cnt}건")

    # 레이블 분포 이상 (카테고리 양성 비율 < 0.5% 또는 > 80%)
    for cat in CATEGORIES:
        if cat not in df.columns:
            continue
        ratio = df[cat].mean()
        if ratio < 0.005:
            issues.append(f"{cat} 양성 비율 {ratio*100:.2f}% (너무 적음)")
        if ratio > 0.80:
            issues.append(f"{cat} 양성 비율 {ratio*100:.1f}% (너무 많음)")

    if issues:
        logger.warning(f"[품질 검증] {label} — 이슈 발견:")
        for iss in issues:
            logger.warning(f"    ⚠  {iss}")
    else:
        logger.info(f"[품질 검증] {label} — 이상 없음 ✓")

    # 기본 통계 출력
    logger.info(f"  행 수: {len(df):,}  |  컬럼: {list(df.columns)}")
    logger.info(f"  텍스트 평균 길이: {df['text'].str.len().mean():.0f}자  "
                f"  최대: {df['text'].str.len().max()}자")


def concat_clean(*dfs) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 3]
    df[CATEGORIES] = df[CATEGORIES].fillna(0).infer_objects(copy=False).astype(int)
    return df


def log_dist(df: pd.DataFrame, label: str) -> None:
    logger.info(f"=== {label} ({len(df):,}건) ===")
    for cat in CATEGORIES:
        n = int(df[cat].sum())
        logger.info(f"  {CATEGORY_KO[cat]:10s} ({cat:20s}): {n:,}건 ({n/len(df)*100:.1f}%)")


def safe_proba(model: Pipeline, texts: pd.Series, cat_idx: int) -> np.ndarray:
    """
    predict_proba는 학습 시 클래스가 1개뿐이면 (n,1) 반환.
    항상 양성 확률을 (n,) 배열로 안전하게 추출.
    """
    proba = model.predict_proba(texts)[cat_idx]
    if proba.shape[1] == 1:
        clf_estimator = model.named_steps["clf"].estimators_[cat_idx]
        fill = float(clf_estimator.classes_[0])
        return np.full(len(texts), fill)
    return proba[:, 1]


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train_full() -> Pipeline:
    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start  = datetime.datetime.now()
    logger.info(f"\n{'='*60}")
    logger.info(f"  RUN ID: {run_id}")
    logger.info(f"{'='*60}")

    # ── 1. 데이터 로드 (Parquet 캐시 우선) ──
    logger.info("\n[1/4] 데이터 로드 (캐시 우선)...")
    bad_safe, bad_unsafe = load_bad()
    # bad_safe/unsafe는 별도 캐싱
    _bad_safe_path = RAW_DIR / "bad_safe.parquet"
    _bad_unsafe_path = RAW_DIR / "bad_unsafe.parquet"
    if not _bad_safe_path.exists():
        bad_safe.to_parquet(_bad_safe_path, index=False)
        bad_unsafe.to_parquet(_bad_unsafe_path, index=False)
        logger.info(f"  BAD 캐시 저장: {RAW_DIR}")
    else:
        bad_safe   = pd.read_parquet(_bad_safe_path)
        bad_unsafe = pd.read_parquet(_bad_unsafe_path)
        logger.info(f"  BAD 캐시 로드: {RAW_DIR}")

    hate       = load_or_cache("hate_speech",  load_hate_speech)
    toxigen    = load_or_cache("toxigen",       load_toxigen)
    hatexplain = load_or_cache("hatexplain",    load_hatexplain)
    ethos      = load_or_cache("ethos",         load_ethos)
    kmhas      = load_or_cache("kmhas",         load_kmhas)
    unsmile    = load_or_cache("unsmile",       load_korean_unsmile)
    jigsaw     = load_or_cache("jigsaw",        load_jigsaw)

    source_counts = {
        "bad_safe":   len(bad_safe),
        "bad_unsafe": len(bad_unsafe),
        "hate":       len(hate),
        "toxigen":    len(toxigen),
        "hatexplain": len(hatexplain),
        "ethos":      len(ethos),
        "kmhas":      len(kmhas),
        "unsmile":    len(unsmile),
        "jigsaw":     len(jigsaw),
    }
    logger.info(f"  소스별 건수: {source_counts}")

    # ── 2. Stage 1 학습 ──
    logger.info("\n[2/4] Stage 1 학습...")
    stage1_df = concat_clean(hate, toxigen, jigsaw, hatexplain, ethos, kmhas, unsmile)

    if len(stage1_df) == 0:
        raise RuntimeError("Stage 1 데이터 없음.")

    # 품질 검증
    validate_dataset(stage1_df, "Stage 1 학습 데이터")
    log_dist(stage1_df, "Stage 1 학습 데이터")

    # 전처리 결과 저장
    stage1_processed_path = PROCESSED_DIR / "stage1_train.parquet"
    stage1_df.to_parquet(stage1_processed_path, index=False)
    logger.info(f"  Stage 1 전처리 데이터 저장: {stage1_processed_path}")

    stage1_model = build_pipeline()

    # 임베딩 캐시 활용
    stage1_embedder = stage1_model.named_steps["emb"]
    stage1_embedder._load_model()
    stage1_texts = list(stage1_df["text"])
    stage1_emb   = embed_with_cache(stage1_embedder, stage1_texts, "stage1")

    # 임베더 없이 LR만 학습
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    stage1_clf = MultiOutputClassifier(LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=42, n_jobs=-1,
    ))
    stage1_clf.fit(stage1_emb, stage1_df[CATEGORIES].values)

    # 검증 F1
    val      = stage1_df.sample(frac=0.1, random_state=42)
    val_idx  = val.index
    val_emb  = stage1_emb[[stage1_df.index.get_loc(i) for i in val_idx]]
    y_pred   = stage1_clf.predict(val_emb)
    stage1_f1 = {}
    logger.info("[Stage 1] 검증 F1:")
    for i, cat in enumerate(CATEGORIES):
        f1v = f1_score(val[CATEGORIES].values[:, i], y_pred[:, i],
                       average="binary", zero_division=0)
        stage1_f1[cat] = round(f1v, 4)
        logger.info(f"  {CATEGORY_KO[cat]:10s}: {f1v:.4f}")

    # ── GPU 메모리 해제 ──
    import torch, gc
    if stage1_embedder._model is not None:
        stage1_embedder._model.cpu()
        del stage1_embedder._model
        stage1_embedder._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"  GPU 메모리 해제: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # ── 3. BAD unsafe 재레이블링 ──
    logger.info(f"\n[3/4] BAD unsafe 재레이블링...")
    bad_unsafe = bad_unsafe.copy()

    # 재레이블링용 임베딩 캐시
    stage1_embedder2 = E5Embedder()
    stage1_embedder2._load_model()
    bad_emb = embed_with_cache(stage1_embedder2, list(bad_unsafe["text"]), "bad_unsafe")

    for i, cat in enumerate(CATEGORIES):
        thr = RELABEL_THR_PER_CAT.get(cat, RELABEL_THR)
        if thr is None:
            bad_unsafe[cat] = 0
            continue
        proba = stage1_clf.estimators_[i].predict_proba(bad_emb)
        if proba.shape[1] == 1:
            bad_unsafe[cat] = 0
        else:
            bad_unsafe[cat] = (proba[:, 1] >= thr).astype(int)
        pos = int(bad_unsafe[cat].sum())
        logger.info(f"  {CATEGORY_KO[cat]:10s} (thr={thr}): {pos:,}건")

    log_dist(bad_unsafe, "BAD unsafe 재레이블링 결과")

    # 재레이블링 결과 저장
    bad_relabeled_path = PROCESSED_DIR / "bad_unsafe_relabeled.parquet"
    bad_unsafe.to_parquet(bad_relabeled_path, index=False)
    logger.info(f"  재레이블링 결과 저장: {bad_relabeled_path}")

    # GPU 해제
    if stage1_embedder2._model is not None:
        stage1_embedder2._model.cpu()
        del stage1_embedder2._model
        stage1_embedder2._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 4. 최종 모델 학습 ──
    logger.info("\n[4/4] 최종 모델 학습...")
    final_df = concat_clean(
        bad_safe, bad_unsafe, hate, toxigen, jigsaw,
        hatexplain, ethos, kmhas, unsmile
    )

    # 품질 검증
    validate_dataset(final_df, "최종 학습 데이터")
    log_dist(final_df, "최종 학습 데이터")

    # 전처리 결과 저장
    final_processed_path = PROCESSED_DIR / "final_train.parquet"
    final_df.to_parquet(final_processed_path, index=False)
    logger.info(f"  최종 학습 데이터 저장: {final_processed_path}")

    test_df  = final_df.sample(frac=0.1, random_state=42)
    train_df = final_df.drop(test_df.index)

    # 임베딩 캐시
    final_embedder = E5Embedder()
    final_embedder._load_model()
    train_emb = embed_with_cache(final_embedder, list(train_df["text"]), "final_train")
    test_emb  = embed_with_cache(final_embedder, list(test_df["text"]),  "final_test")

    final_clf = MultiOutputClassifier(LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=42, n_jobs=-1,
    ))
    final_clf.fit(train_emb, train_df[CATEGORIES].values)

    y_pred_final = final_clf.predict(test_emb)
    final_f1 = {}
    logger.info("[최종 모델] Test Set F1:")
    for i, cat in enumerate(CATEGORIES):
        f1v = f1_score(test_df[CATEGORIES].values[:, i], y_pred_final[:, i],
                       average="binary", zero_division=0)
        final_f1[cat] = round(f1v, 4)
        logger.info(f"  {CATEGORY_KO[cat]:10s}: {f1v:.4f}")

    # Pipeline으로 재조립 (저장/서빙용)
    final_model = Pipeline([
        ("emb", final_embedder),
        ("clf", final_clf),
    ])

    # ── 메타데이터 저장 ──
    elapsed = (datetime.datetime.now() - t_start).seconds
    run_info = {
        "run_id":          run_id,
        "timestamp":       t_start.isoformat(),
        "elapsed_sec":     elapsed,
        "model":           "multilingual-e5-small + LogisticRegression",
        "sources":         source_counts,
        "total_train":     len(train_df),
        "total_test":      len(test_df),
        "thresholds":      RELABEL_THR_PER_CAT,
        "stage1_f1":       stage1_f1,
        "final_f1":        final_f1,
        "data_paths": {
            "stage1":      str(stage1_processed_path),
            "relabeled":   str(bad_relabeled_path),
            "final":       str(final_processed_path),
        }
    }
    save_metadata(run_info)
    logger.info(f"\n  총 소요 시간: {elapsed}초")

    return final_model


# ──────────────────────────────────────────────
# 저장 / 로드
# ──────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"모델 저장: {path}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    import torch
    logger.info(f"모델 로드: {path}")
    # GPU OOM 방지: map_location으로 CPU에 먼저 로드
    import io
    class CpuUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(module, name)
    # torch.load의 map_location을 cpu로 강제
    _orig = torch.load
    torch.load = lambda f, **kw: _orig(f, map_location="cpu", **{k:v for k,v in kw.items() if k != "map_location"})
    try:
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
    finally:
        torch.load = _orig
    # 로드 후 GPU로 이동 시도
    emb = pipeline.named_steps.get("emb")
    if emb is not None and emb._model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            emb._model = emb._model.to(device)
            emb._batch_size = 32 if device == "cuda" else 64
        except RuntimeError:
            logger.warning("GPU OOM → CPU로 폴백")
    return pipeline


# ──────────────────────────────────────────────
# 필터링
# ──────────────────────────────────────────────

def filter_text_v2(text: str, pipeline: Pipeline) -> dict:
    """
    개선된 필터링 (SemEval-2021 Task 5: Toxic Spans Detection 참고).
    Pavlopoulos et al. (2021) — 문장 전체 독성 판단이 아닌
    어떤 span이 독성을 유발하는지 탐지하는 방식에서 착안.

    논문 방식: 토큰 레벨 BIO 태깅 (BERT 기반)
    본 구현:  문장 레벨 분류기 + 단어 패턴 매칭으로 경량화 적용
      1. 전체 텍스트를 문맥으로 읽어서 유해 여부 판단 (맥락 기반)
      2. 문장 단위로 분리해서 유해 문장 탐지 (toxic span 개념)
      3. 유해 문장 전체를 단어별 * 마스킹 (씨발→**, fuck→****)

    Reference:
      Pavlopoulos et al. (2021). SemEval-2021 Task 5: Toxic Spans Detection.
      https://aclanthology.org/2021.semeval-1.6
    """
    import re as _re

    _PROF_PATTERNS = [
        r"[ㅅㅆ][ㅣi][ㅂ발]", r"ㅅㅂ", r"ㅆㅂ", r"미친", r"ㅁㅊ",
        r"멍[청충]", r"씨[발팔]", r"개[새색]끼", r"병신", r"ㅂㅅ",
        r"존나", r"ㅈㄴ", r"[ㄱ개][ㅅ새][ㄲ끼]",
        r"fuck", r"shit", r"bitch", r"asshole", r"idiot", r"stupid",
    ]

    def _is_profane_word(w):
        w = w.lower().strip()
        return any(_re.search(p, w, _re.IGNORECASE) for p in _PROF_PATTERNS)

    def _partial_mask(w):
        return "*" * len(w)

    # ── 1. 전체 텍스트 문맥 판단 ──
    scores = {cat: float(safe_proba(pipeline, pd.Series([text]), i)[0])
              for i, cat in enumerate(CATEGORIES)}
    detected = [cat for cat, s in scores.items() if s >= FILTER_THR]
    blocked  = len(detected) > 0

    if not blocked:
        return {
            "input": text, "scores": {c: round(s,4) for c,s in scores.items()},
            "detected": [], "output": text, "blocked": False,
            "masked_sentences": []
        }

    # ── 2. 문장 단위 분리 후 유해 문장 차단 ──
    sentences = _re.split(r'(?<=[.!?])\s*', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]

    masked_sentences = []
    output_parts = []

    for sent in sentences:
        if not sent.strip():
            output_parts.append(sent)
            continue

        # 문장별 유해 판단
        s_scores = [float(safe_proba(pipeline, pd.Series([sent]), i)[0])
                    for i in range(len(CATEGORIES))]
        cat_thrs = [FILTER_THR_PER_CAT.get(CATEGORIES[i], FILTER_THR)
                    for i in range(len(CATEGORIES))]
        s_blocked = any(s >= t for s, t in zip(s_scores, cat_thrs))

        if s_blocked:
            # 유해 카테고리 목록
            cats = [CATEGORY_KO[CATEGORIES[i]] for i, (s, t) in
                    enumerate(zip(s_scores, cat_thrs)) if s >= t]
            masked_sentences.append(sent.strip())
            output_parts.append(f"[{'/'.join(cats)} 차단]")
        else:
            output_parts.append(sent)

    output = ' '.join(output_parts)
    return {
        "input":            text,
        "scores":           {c: round(s,4) for c,s in scores.items()},
        "detected":         detected,
        "output":           output,
        "blocked":          blocked,
        "masked_sentences": masked_sentences,
    }

def filter_text(text: str, pipeline: Pipeline) -> dict:
    scores   = {cat: float(safe_proba(pipeline, pd.Series([text]), i)[0])
                for i, cat in enumerate(CATEGORIES)}
    detected = [cat for cat, s in scores.items() if s >= FILTER_THR]
    blocked  = len(detected) > 0

    if blocked:
        words  = text.split()
        masked = []
        for word in words:
            w_scores = [float(safe_proba(pipeline, pd.Series([word]), i)[0])
                        for i in range(len(CATEGORIES))]
            masked.append("*" * len(word) if max(w_scores) >= FILTER_THR else word)
        output = " ".join(masked)
    else:
        output = text

    return {
        "input":    text,
        "scores":   {cat: round(s, 4) for cat, s in scores.items()},
        "detected": detected,
        "output":   output,
        "blocked":  blocked,
    }


def print_result(result: dict) -> None:
    bar = "─" * 54
    print(f"\n{bar}")
    print(f"  입력  : {result['input']}")
    print(f"  출력  : {result['output']}")
    print(f"  차단  : {'예' if result['blocked'] else '아니오'}")
    if result["detected"]:
        print(f"  감지  : {' / '.join(CATEGORY_KO[c] for c in result['detected'])}")
    print(f"  점수  :")
    for cat, score in result["scores"].items():
        flag = " ◀" if cat in result["detected"] else ""
        print(f"    {CATEGORY_KO[cat]:10s}: {score:.4f}{flag}")
    print(bar)


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-label Toxic Filter v2")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--filter",      type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.train:
        pipeline = train_full()
        save_model(pipeline)
        return

    if not Path(MODEL_PATH).exists():
        logger.error(f"모델 없음: {MODEL_PATH}  →  먼저 --train 실행")
        return

    pipeline = load_model()
    logger.info(f"모델 로드: {MODEL_PATH}")

    if args.filter:
        print_result(filter_text_v2(args.filter, pipeline))
        return

    if args.interactive:
        print(f"다중 레이블 필터 v2  |  {', '.join(CATEGORY_KO.values())}")
        print("종료: q\n")
        while True:
            try:
                text = input("입력 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            print_result(filter_text_v2(text, pipeline))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────
# 개선된 필터링: 문맥 기반 + 부분 마스킹
# ──────────────────────────────────────────────

import re as _re

# 욕설 패턴 사전 (앞글자 보존 마스킹용)
PROFANITY_PATTERNS = [
    # 한국어 욕설 패턴
    r'[ㅅㅆ][ㅣi][ㅂ발]', r'ㅅㅂ', r'ㅆㅂ',
    r'[ㅁㅂ][ㅓㅔ][ㄴ]', r'ㅁㅊ', r'미친',
    r'[ㅈㅉ]같', r'ㅈㄴ', r'존나',
    r'[ㄱ개][ㅅ새][ㄲ끼]', r'병신', r'ㅂㅅ',
    r'씨[발팔]', r'개[새색]끼',
    r'멍[청충]', r'바보', r'멍텅구리',
    r'말이\s*줄줄', r'줄줄\s*새',  # 간접 표현
    # 영어 욕설
    r'fuck', r'shit', r'bitch', r'asshole', r'bastard',
    r'idiot', r'stupid', r'moron', r'dumb',
]

def _partial_mask(word: str) -> str:
    """단어 전체를 * 로 마스킹. 씨발 → **, fuck → ****"""
    return '*' * len(word)

def _is_profane_word(word: str) -> bool:
    """패턴 매칭으로 욕설 단어 여부 판단"""
    w = word.lower().strip()
    for pattern in PROFANITY_PATTERNS:
        if _re.search(pattern, w, _re.IGNORECASE):
            return True
    return False


