"""
tune_threshold.py
=================
카테고리별 최적 threshold 탐색 (과잉 예측 → precision 회복).

방법:
  1. val set에서 각 카테고리별 F1 최대화하는 threshold 탐색 (0.1~0.9)
  2. 찾은 threshold를 test set에 적용 → 정직한 평가 (과적합 방지)
  3. 0.5 고정 vs 튜닝 threshold 비교

LR / MLP 둘 다 지원. E5 임베딩(384-dim) 통합 모델 기준.

사용:
  python tune_threshold.py                    # MLP
  python tune_threshold.py --model lr
  python tune_threshold.py --balanced --min-support 50
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH     = Path("data/llm_dataset/dataset.sqlite")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
EMBED_DIM   = 384

CATEGORIES = ["profanity", "hate_speech", "gender", "threat", "political", "other"]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech": "혐오발언", "gender": "성 관련",
    "threat": "살해협박", "political": "정치", "other": "기타유해",
}


def load_split(conn, split):
    cat_cols = ", ".join(f"l.{c}" for c in CATEGORIES)
    sql = f"""
        SELECT t.id, t.lang, e.vector, {cat_cols}
        FROM texts t
        JOIN splits s     ON t.id = s.text_id
        JOIN embeddings e ON t.id = e.text_id
        JOIN labels l     ON t.id = l.text_id
        WHERE s.split = ?
    """
    rows = conn.execute(sql, (split,)).fetchall()
    if not rows:
        return None, None, None
    langs = [r[1] for r in rows]
    X = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    Y = np.array([r[3:] for r in rows], dtype=np.int8)
    return X, Y, langs


def get_proba_lr(path, X):
    """LR predict_proba → (n, n_cat) 양성 확률"""
    with open(path, "rb") as f:
        model = pickle.load(f)["model"]
    proba = np.zeros((len(X), len(CATEGORIES)), dtype=np.float32)
    for i, est in enumerate(model.estimators_):
        p = est.predict_proba(X)
        proba[:, i] = p[:, 1] if p.shape[1] == 2 else p[:, 0]
    return proba


def get_proba_mlp(path, X):
    """MLP sigmoid → (n, n_cat) 양성 확률"""
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dim=EMBED_DIM, hidden=(256, 128), out_dim=len(CATEGORIES), dropout=0.2):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers += [nn.Linear(prev, out_dim)]
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    bundle = torch.load(path, weights_only=False)
    model = MLP()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        proba = torch.sigmoid(logits).numpy()
    return proba


def f1_at(y_true, proba, thr):
    from sklearn.metrics import f1_score
    pred = (proba >= thr).astype(np.int8)
    return f1_score(y_true, pred, average="binary", zero_division=0)


def find_best_threshold(y_true_col, proba_col):
    """val에서 F1 최대화하는 threshold 탐색 (0.10~0.90, 0.02 간격)"""
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.10, 0.91, 0.02):
        f1 = f1_at(y_true_col, proba_col, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def evaluate_with_thresholds(Y, proba, thresholds, langs=None, min_support=50):
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    pred = np.zeros_like(proba, dtype=np.int8)
    for i in range(len(CATEGORIES)):
        pred[:, i] = (proba[:, i] >= thresholds[i]).astype(np.int8)

    per_cat = []
    f1s = []
    for i, c in enumerate(CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            Y[:, i], pred[:, i], average="binary", zero_division=0)
        sup = int(Y[:, i].sum())
        per_cat.append((c, p, r, f, sup, thresholds[i]))
        if sup >= min_support:
            f1s.append(f)
    macro = float(np.mean(f1s)) if f1s else 0.0
    micro = f1_score(Y, pred, average="micro", zero_division=0)

    lang_macro = {}
    if langs is not None:
        langs = np.array(langs)
        for lg in ["en", "ko"]:
            mask = langs == lg
            if mask.sum() == 0:
                continue
            lf1 = []
            for i in range(len(CATEGORIES)):
                if Y[mask, i].sum() >= min_support:
                    lf1.append(f1_score(Y[mask, i], pred[mask, i], zero_division=0))
            lang_macro[lg] = float(np.mean(lf1)) if lf1 else 0.0
    return per_cat, macro, micro, lang_macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "mlp"], default="mlp")
    ap.add_argument("--min-support", type=int, default=50)
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info(f"[Threshold 튜닝] {args.model.upper()}")
    logger.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    logger.info("  val/test 로드 중...")
    Xv, Yv, _      = load_split(conn, "val")
    Xt, Yt, langs  = load_split(conn, "test")
    conn.close()
    logger.info(f"  val: {Xv.shape}, test: {Xt.shape}")

    # 확률 예측
    if args.model == "lr":
        proba_v = get_proba_lr(MODELS_DIR / "lr_model.pkl", Xv)
        proba_t = get_proba_lr(MODELS_DIR / "lr_model.pkl", Xt)
    else:
        proba_v = get_proba_mlp(MODELS_DIR / "mlp_model.pt", Xv)
        proba_t = get_proba_mlp(MODELS_DIR / "mlp_model.pt", Xt)

    # ── 1. val에서 카테고리별 최적 threshold 탐색 ──
    logger.info("\n  ━━━ val에서 최적 threshold 탐색 ━━━")
    best_thr = {}
    for i, c in enumerate(CATEGORIES):
        thr, f1 = find_best_threshold(Yv[:, i], proba_v[:, i])
        best_thr[i] = thr
        logger.info(f"  {CATEGORY_KO[c]:10s}: thr={thr:.2f}  (val F1={f1:.3f})")

    thr_tuned = [best_thr[i] for i in range(len(CATEGORIES))]
    thr_fixed = [0.5] * len(CATEGORIES)

    # ── 2. test에 적용 — 0.5 고정 vs 튜닝 비교 ──
    logger.info("\n  ━━━ test 평가: 0.5 고정 ━━━")
    pc_f, macro_f, micro_f, lang_f = evaluate_with_thresholds(Yt, proba_t, thr_fixed, langs, args.min_support)
    logger.info(f"  {'카테고리':10s} {'P':>6s} {'R':>6s} {'F1':>6s}  {'Support':>8s}")
    for c, p, r, f, sup, _ in pc_f:
        logger.info(f"  {CATEGORY_KO[c]:10s} {p:6.3f} {r:6.3f} {f:6.3f}  {sup:8d}")
    logger.info(f"  macro-F1: {macro_f:.4f} | micro-F1: {micro_f:.4f}")
    if lang_f:
        logger.info(f"  언어별: en={lang_f.get('en',0):.4f}  ko={lang_f.get('ko',0):.4f}")

    logger.info("\n  ━━━ test 평가: 튜닝 threshold ━━━")
    pc_t, macro_t, micro_t, lang_t = evaluate_with_thresholds(Yt, proba_t, thr_tuned, langs, args.min_support)
    logger.info(f"  {'카테고리':10s} {'P':>6s} {'R':>6s} {'F1':>6s}  {'Support':>8s} {'thr':>5s}")
    for c, p, r, f, sup, th in pc_t:
        logger.info(f"  {CATEGORY_KO[c]:10s} {p:6.3f} {r:6.3f} {f:6.3f}  {sup:8d} {th:5.2f}")
    logger.info(f"  macro-F1: {macro_t:.4f} | micro-F1: {micro_t:.4f}")
    if lang_t:
        logger.info(f"  언어별: en={lang_t.get('en',0):.4f}  ko={lang_t.get('ko',0):.4f}")

    # ── 3. 요약 ──
    logger.info("\n" + "=" * 60)
    logger.info("  개선 요약")
    logger.info("=" * 60)
    logger.info(f"  macro-F1: {macro_f:.4f} → {macro_t:.4f}  ({macro_t-macro_f:+.4f})")
    logger.info(f"  micro-F1: {micro_f:.4f} → {micro_t:.4f}  ({micro_t-micro_f:+.4f})")

    # threshold 저장 (classify.py에서 쓸 수 있게)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {CATEGORIES[i]: thr_tuned[i] for i in range(len(CATEGORIES))}
    with open(RESULTS_DIR / f"best_thresholds_{args.model}.json", "w") as f:
        json.dump({"model": args.model, "thresholds": out,
                   "macro_f1_fixed": macro_f, "macro_f1_tuned": macro_t}, f, indent=2)
    logger.info(f"\n  threshold 저장: {RESULTS_DIR / 'best_thresholds.json'}")


if __name__ == "__main__":
    main()
