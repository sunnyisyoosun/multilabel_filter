"""
multilabel_filter.py
=======================
2단계 학습 파이프라인 (데이터 로더 + CATEGORIES 정의).
※ 본 파일은 직접 실행하지 않고, prepare_llm_dataset_v3.py가 load 함수들과
  CATEGORIES를 import해서 사용한다.

카테고리 (6개, 다중 레이블):
    profanity   - 욕설
    hate_speech - 혐오발언 (인종/종교/지역/장애/성소수자)
    gender      - 성 관련 (성차별 + 성희롱 통합, 여성/남성)
    threat      - 살해 협박
    political   - 정치
    other       - 기타유해 (직업/나이/기타)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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

CATEGORIES = ["profanity", "hate_speech", "gender", "threat", "political", "other"]
CATEGORY_KO = {
    "profanity":   "욕설",
    "hate_speech": "혐오발언",
    "gender":      "성 관련",
    "threat":      "살해 협박",
    "political":   "정치",
    "other":       "기타유해",
}
MODEL_PATH  = "multilabel_filter.pkl"
RELABEL_THR = 0.50
RELABEL_THR_PER_CAT = {
    "profanity":   0.55,
    "hate_speech": 0.55,
    "gender":      0.50,
    "threat":      0.75,
    "political":   0.55,
    "other":       0.55,
}
FILTER_THR  = 0.40
FILTER_THR_PER_CAT = {
    "profanity":   0.40,
    "hate_speech": 0.45,
    "gender":      0.50,
    "threat":      0.45,
    "political":   0.50,
    "other":       0.50,
}

# ── 데이터 엔지니어링 경로 ──
DATA_DIR       = Path("data")
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EMB_DIR        = DATA_DIR / "embeddings"
META_DIR       = DATA_DIR / "metadata"
for _d in [RAW_DIR, PROCESSED_DIR, EMB_DIR, META_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def _ensure_categories(df: pd.DataFrame) -> pd.DataFrame:
    """CATEGORIES 컬럼이 없으면 0으로 채움 (KeyError 방지)."""
    for c in CATEGORIES:
        if c not in df.columns:
            df[c] = 0
    return df


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_bad() -> tuple:
    """BAD → safe / unsafe 분리."""
    logger.info("BAD 로드 중 (HuggingFace)...")
    try:
        from datasets import load_dataset
        safe_rows, unsafe_rows = [], []
        for split in ["train", "validation", "test"]:
            try:
                ds = load_dataset("facebook/bot_adversarial_dialogues", split=split)
            except Exception:
                ds = load_dataset("allenai/bot_adversarial_dialogue", split=split)
            df = ds.to_pandas()
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
        import os, tensorflow_datasets as tfds
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    jigsaw_toxicity — threat 전용 소스 (threat==1만 추출).
    obscene 매핑 제거 (음란물은 우리 카테고리와 안 맞음).
    """
    logger.info("jigsaw_toxicity 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/jigsaw_toxicity", split="train")
        df = ds.to_pandas()
        logger.info(f"  jigsaw 컬럼: {list(df.columns)}")
        text_col = next((c for c in ["comment_text", "text", "comment"] if c in df.columns), None)
        if text_col is None:
            raise KeyError(f"텍스트 컬럼 없음. 실제: {list(df.columns)}")
        df["text"]        = df[text_col].astype(str)
        df["profanity"]   = df.get("toxic",         pd.Series(0, index=df.index)).astype(int)
        df["hate_speech"] = df.get("identity_hate", pd.Series(0, index=df.index)).astype(int)
        df["gender"]      = 0
        df["threat"]      = df.get("threat",        pd.Series(0, index=df.index)).astype(int)
        df["political"]   = 0
        df["other"]       = 0
        df = _ensure_categories(df)
        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        result = result[result["threat"] == 1].reset_index(drop=True)
        logger.info(f"  jigsaw: {len(result):,}건  (threat 전용)")
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
        df["text"]        = df["text"].astype(str)
        df["profanity"]   = (df["class"] == 1).astype(int)
        df["hate_speech"] = (df["class"] == 0).astype(int)
        df["gender"]      = 0
        df["threat"]      = 0
        df["political"]   = 0
        df["other"]       = 0
        df = _ensure_categories(df)
        result = df[["text"] + CATEGORIES]
        logger.info(f"  hate_speech_offensive: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  hate_speech_offensive 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_toxigen() -> pd.DataFrame:
    """
    toxigen-data: 텍스트 컬럼 'generation'.
    women → gender / lgbtq·trans → hate_speech(성소수자=정체성 혐오) / race → hate_speech
    """
    logger.info("toxigen 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("toxigen/toxigen-data", split="train")
        df = ds.to_pandas()
        logger.info(f"  toxigen 실제 컬럼: {list(df.columns)}")
        for col in ["generation", "text", "prompt"]:
            if col in df.columns:
                df["text"] = df[col].astype(str)
                break
        else:
            raise KeyError(f"텍스트 컬럼 없음. 실제 컬럼: {list(df.columns)}")
        tox_col = next((c for c in ["toxicity_human", "prompt_label", "label"] if c in df.columns), None)
        if tox_col:
            df["is_toxic"] = (pd.to_numeric(df[tox_col], errors="coerce").fillna(0) >= 0.5).astype(int)
        else:
            df["is_toxic"] = 0

        RACE_GROUPS   = {"asian", "black", "latino", "native_american", "middle_east", "jewish"}
        LGBTQ_GROUPS  = {"lgbtq", "trans"}     # 성소수자 → hate_speech
        GENDER_GROUPS = {"women"}              # 여성 → gender

        def map_labels(row):
            grp = str(row.get("target_group", "")).lower()
            tox = row["is_toxic"]
            is_race  = any(g in grp for g in RACE_GROUPS)
            is_lgbtq = any(g in grp for g in LGBTQ_GROUPS)
            is_women = any(g in grp for g in GENDER_GROUPS)
            return {
                "profanity":   0,
                "hate_speech": int(tox and (is_race or is_lgbtq)),
                "gender":      int(tox and is_women),
                "threat":      0,
                "political":   0,
                "other":       0,
            }

        mapped = df.apply(map_labels, axis=1, result_type="expand")
        result = pd.concat([df[["text"]], mapped], axis=1)
        result = _ensure_categories(result)[["text"] + CATEGORIES]
        logger.info(f"  toxigen: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  toxigen 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_hatexplain() -> pd.DataFrame:
    """
    HateXplain — GitHub raw JSON.
    hate=0 → profanity+hate_speech / offensive=2 → profanity / violence rationale → threat
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
                labels = [a["label"] for a in item.get("annotators", [])]
                if not labels:
                    continue
                majority = Counter(labels).most_common(1)[0][0]
                tokens = item.get("post_tokens", [])
                text = " ".join(tokens) if tokens else ""
                if len(text.strip()) < 5:
                    continue
                profanity   = int(majority in [0, 2])
                hate_speech = int(majority == 0)
                rationales = item.get("rationales", [])
                has_violence = int(any("violence" in str(r).lower() for r in rationales))
                rows.append({
                    "text": text,
                    "profanity": profanity,
                    "hate_speech": hate_speech,
                    "gender": 0,
                    "threat": has_violence,
                    "political": 0,
                    "other": 0,
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        df = _ensure_categories(df)
        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  hatexplain: {len(result):,}건  (threat 양성: {int(result['threat'].sum())}건)")
        return result
    except Exception as e:
        logger.warning(f"  hatexplain 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_ethos() -> pd.DataFrame:
    """
    ETHOS multilabel — Parquet 직접 로드.
    violence → threat / gender → gender / sexual_orientation → hate_speech / race·national_origin → hate_speech
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

        df["profanity"]   = to_int_col("directed_vs_generalized")
        df["hate_speech"] = (to_int_col("race") + to_int_col("national_origin")
                             + to_int_col("religion") + to_int_col("disability")
                             + to_int_col("sexual_orientation")).clip(0, 1)
        df["gender"]      = to_int_col("gender")
        df["threat"]      = to_int_col("violence")
        df["political"]   = 0
        df["other"]       = 0
        df = _ensure_categories(df)
        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  ethos: {len(result):,}건  (threat 양성: {int(result['threat'].sum())}건)")
        return result
    except Exception as e:
        logger.warning(f"  ethos 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_khaters() -> pd.DataFrame:
    """
    K-HATERS (humane-lab/K-HATERS) — 172K 한국어 뉴스 댓글, 타겟별 라벨.
    label: normal / offensive / L1_hate / L2_hate
    target_label(배열): individual/political/region/others/job/gender/age/disabled/religion

    매핑:
      normal                              → 전부 0
      target gender                       → gender (성차별+성희롱 통합)
      target region/religion/disabled     → hate_speech
      target political                    → political
      target others/job/age               → other
      target individual/none + offensive  → profanity
    """
    logger.info("K-HATERS (한국어) 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("humane-lab/K-HATERS", split="train")
        df = ds.to_pandas()
        logger.info(f"  K-HATERS 컬럼: {list(df.columns)}")
        df["text"] = df["text"].astype(str)

        HATE_TARGETS = {"region", "religion", "disabled"}

        def map_row(row):
            label = str(row.get("label", "normal"))
            targets_raw = row.get("target_label", [])
            if hasattr(targets_raw, "__iter__") and not isinstance(targets_raw, str):
                targets = set(str(t) for t in targets_raw)
            else:
                targets = {str(targets_raw)}

            out = {c: 0 for c in CATEGORIES}
            if label == "normal":
                return out  # 정상

            is_hate = label in ("L1_hate", "L2_hate")
            is_off  = label == "offensive"

            # 타겟 기준 매핑
            if "gender" in targets:
                out["gender"] = 1
            if targets & HATE_TARGETS:
                out["hate_speech"] = 1
            if "political" in targets:
                out["political"] = 1
            if targets & {"others", "job", "age"}:
                out["other"] = 1
            # 개인/불특정 공격 → 욕설
            if (targets & {"individual", "none"}) or not targets:
                out["profanity"] = 1
            # 아무것도 안 잡혔으면 (예: offensive인데 타겟이 정체성) 욕설로 보정
            if sum(out.values()) == 0 and (is_hate or is_off):
                out["profanity"] = 1
            return out

        mapped = df.apply(map_row, axis=1, result_type="expand")
        result = pd.concat([df[["text"]], mapped], axis=1)
        result = _ensure_categories(result)[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  K-HATERS: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  K-HATERS 로드 최종 실패: {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


def load_korean_unsmile() -> pd.DataFrame:
    """
    Korean UnSmile — 한국어 혐오 멀티레이블.
    악플/욕설 → profanity
    여성/가족, 남성 → gender (성차별+성희롱 통합)
    성소수자/인종·국적/연령/지역/종교 → hate_speech
    기타 혐오 → other
    """
    logger.info("Korean UnSmile 로드 중...")
    try:
        from datasets import load_dataset
        ds = load_dataset("smilegate-ai/kor_unsmile", split="train")
        df = ds.to_pandas()
        logger.info(f"  UnSmile 컬럼: {list(df.columns)}")
        df["text"] = df.get("문장", df.get("text", pd.Series("", index=df.index))).astype(str)

        def col(name):
            return df.get(name, pd.Series(0, index=df.index)).fillna(0).astype(float)

        df["profanity"]   = (col("악플/욕설") >= 0.5).astype(int)
        df["hate_speech"] = ((col("성소수자") + col("인종/국적")
                              + col("연령") + col("지역") + col("종교")) >= 0.5).clip(0, 1).astype(int)
        df["gender"]      = ((col("여성/가족") + col("남성")) >= 0.5).clip(0, 1).astype(int)
        df["threat"]      = 0
        df["political"]   = 0
        df["other"]       = (col("기타 혐오") >= 0.5).astype(int)
        df = _ensure_categories(df)
        result = df[["text"] + CATEGORIES].dropna(subset=["text"])
        logger.info(f"  Korean UnSmile: {len(result):,}건")
        return result
    except Exception as e:
        logger.warning(f"  Korean UnSmile 로드 실패 (건너뜀): {e}")
        return pd.DataFrame(columns=["text"] + CATEGORIES)


# ──────────────────────────────────────────────
# 유틸 (E5 임베더 등) — 직접 학습용, 현 파이프라인에선 미사용
# ──────────────────────────────────────────────

class E5Embedder(BaseEstimator, TransformerMixin):
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
                device = "cuda"; self._batch_size = self.batch_size or 32
            else:
                device = "cpu"; self._batch_size = self.batch_size or 64
            logger.info(f"  E5 모델 로드 중: {self.model_name} (device={device})")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit(self, X, y=None):
        self._load_model(); return self

    def transform(self, X):
        import torch
        model = self._load_model()
        texts = list(X) if not isinstance(X, list) else X
        texts = [f"passage: {t}" for t in texts]
        CHUNK = 10000
        all_emb = []
        for start in range(0, len(texts), CHUNK):
            chunk = texts[start:start + CHUNK]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                emb = model.encode(chunk, batch_size=self._batch_size,
                                   show_progress_bar=len(texts) > 1000,
                                   normalize_embeddings=True)
            all_emb.append(emb)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.vstack(all_emb).astype("float32")


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.parquet"

def load_or_cache(name: str, loader_fn) -> pd.DataFrame:
    """Parquet 캐시가 있으면 로드, 없으면 loader_fn() 실행 후 저장."""
    path = _cache_path(name)
    if path.exists():
        logger.info(f"  [{name}] 캐시 로드: {path}")
        return pd.read_parquet(path)
    df = loader_fn()
    if df is not None and len(df) > 0:
        df.to_parquet(path, index=False)
        logger.info(f"  [{name}] Parquet 저장: {path}  ({len(df):,}건)")
    return df


def concat_clean(*dfs) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 3]
    for c in CATEGORIES:
        if c not in df.columns:
            df[c] = 0
    df[CATEGORIES] = df[CATEGORIES].fillna(0).infer_objects(copy=False).astype(int)
    return df


def log_dist(df: pd.DataFrame, label: str) -> None:
    logger.info(f"=== {label} ({len(df):,}건) ===")
    for cat in CATEGORIES:
        n = int(df[cat].sum())
        logger.info(f"  {CATEGORY_KO[cat]:10s} ({cat:20s}): {n:,}건 ({n/len(df)*100:.1f}%)")


# ──────────────────────────────────────────────
# 진입점 (직접 실행 시 데이터 로드 점검용)
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-label Toxic Filter (6 categories)")
    parser.add_argument("--check", action="store_true", help="데이터 로드 점검")
    args = parser.parse_args()

    if args.check:
        logger.info("데이터 로드 점검 모드")
        loaders = [
            ("hate_speech", load_hate_speech), ("toxigen", load_toxigen),
            ("hatexplain", load_hatexplain), ("ethos", load_ethos),
            ("jigsaw", load_jigsaw), ("khaters", load_khaters),
            ("unsmile", load_korean_unsmile),
        ]
        for name, fn in loaders:
            df = fn()
            if df is not None and len(df) > 0:
                log_dist(df, name)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
