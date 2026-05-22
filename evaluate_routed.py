"""
evaluate_routed.py
==================
언어별 라우팅 구조의 통합 평가.
  영어 test → 영어 분류기 (E5 임베딩, lr_model_en/mlp_model_en)
  한국어 test → 한국어 분류기 (KcELECTRA 임베딩, lr_model_ko/mlp_model_ko)
  → 두 결과를 합쳐서 전체 성능 산출 + 언어별 비교

출력:
  results/routed_metrics.json
  results/routed_comparison.csv

사용:
  python evaluate_routed.py
  python evaluate_routed.py --min-support 50
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

E5_DIM = 384
KO_DIM = 768

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech": "혐오발언", "sexual_harassment": "성희롱",
    "sexism": "성차별", "threat": "살해협박", "political": "정치", "other": "기타유해",
}


# ──────────────────────────────────────────────
# 데이터 로드 (언어별, 다른 임베딩 테이블)
# ──────────────────────────────────────────────

def load_test_lang(conn, lang: str, embed_table: str):
    """특정 언어의 test 데이터를 지정 임베딩 테이블에서 로드"""
    cat_cols = ", ".join(f"l.{c}" for c in CATEGORIES)
    sql = f"""
        SELECT t.id, e.vector, {cat_cols}
        FROM texts t
        JOIN splits s     ON t.id = s.text_id
        JOIN {embed_table} e ON t.id = e.text_id
        JOIN labels l     ON t.id = l.text_id
        WHERE s.split = 'test' AND t.lang = ?
    """
    rows = conn.execute(sql, (lang,)).fetchall()
    if not rows:
        return None, None
    dim = E5_DIM if embed_table == "embeddings" else KO_DIM
    X = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    Y = np.array([r[2:] for r in rows], dtype=np.int8)
    return X, Y


# ──────────────────────────────────────────────
# 모델 로드 + 예측
# ──────────────────────────────────────────────

def predict_lr(path: Path, X):
    with open(path, "rb") as f:
        model = pickle.load(f)["model"]
    return model.predict(X).astype(np.int8)


def predict_mlp(path: Path, X, in_dim: int):
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, len(CATEGORIES)),
            )
        def forward(self, x):
            return self.net(x)

    bundle = torch.load(path, weights_only=False)
    model = MLP()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        pred = (torch.sigmoid(logits) > 0.5).numpy().astype(np.int8)
    return pred


# ──────────────────────────────────────────────
# 메트릭
# ──────────────────────────────────────────────

def compute_metrics(Y_true, Y_pred, min_support=0):
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    per_cat = []
    f1s = []
    eligible = []
    for i, c in enumerate(CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            Y_true[:, i], Y_pred[:, i], average="binary", zero_division=0)
        sup = int(Y_true[:, i].sum())
        per_cat.append({"category": c, "category_ko": CATEGORY_KO[c],
                        "precision": float(p), "recall": float(r), "f1": float(f),
                        "support": sup, "eligible": sup >= min_support})
        f1s.append(float(f))
        if sup >= min_support:
            eligible.append(i)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    weighted = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
    macro_filt = float(np.mean([f1s[i] for i in eligible])) if (min_support > 0 and eligible) else None
    return {"per_category": per_cat, "macro_f1": float(macro), "micro_f1": float(micro),
            "weighted_f1": float(weighted), "macro_f1_filtered": macro_filt,
            "eligible_categories": [CATEGORIES[i] for i in eligible]}


def print_metrics(metrics, title, min_support):
    logger.info(f"\n  ━━━ {title} ━━━")
    logger.info(f"  {'카테고리':12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Support':>9s}")
    for r in metrics["per_category"]:
        mark = "" if r.get("eligible", True) else " (x)"
        logger.info(f"  {r['category_ko']:12s} {r['precision']:7.3f} {r['recall']:7.3f} "
                    f"{r['f1']:7.3f} {r['support']:9d}{mark}")
    logger.info(f"  {'─'*50}")
    logger.info(f"  macro-F1   : {metrics['macro_f1']:.4f}")
    logger.info(f"  micro-F1   : {metrics['micro_f1']:.4f}")
    logger.info(f"  weighted-F1: {metrics['weighted_f1']:.4f}")
    if metrics.get("macro_f1_filtered") is not None:
        logger.info(f"  macro-F1 (support>={min_support}): {metrics['macro_f1_filtered']:.4f}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "mlp"], default="mlp")
    ap.add_argument("--min-support", type=int, default=50)
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info(f"[평가] 언어 라우팅 구조 ({args.model.upper()})")
    logger.info("=" * 60)

    if not DB_PATH.exists():
        logger.error(f"  DB 없음: {DB_PATH}")
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # ── 영어: E5 임베딩 + 영어 모델 ──
    logger.info("\n  [영어] E5 임베딩 + 영어 분류기")
    Xe, Ye = load_test_lang(conn, "en", "embeddings")
    if Xe is None:
        logger.error("  영어 test 데이터 없음")
        return
    logger.info(f"    test: {Xe.shape}")
    if args.model == "lr":
        pred_en = predict_lr(MODELS_DIR / "lr_model_en.pkl", Xe)
    else:
        pred_en = predict_mlp(MODELS_DIR / "mlp_model_en.pt", Xe, E5_DIM)
    metrics_en = compute_metrics(Ye, pred_en, args.min_support)
    print_metrics(metrics_en, "영어 (E5)", args.min_support)

    # ── 한국어: KcELECTRA 임베딩 + 한국어 모델 ──
    logger.info("\n  [한국어] KcELECTRA 임베딩 + 한국어 분류기")
    Xk, Yk = load_test_lang(conn, "ko", "embeddings_ko")
    if Xk is None:
        logger.error("  한국어 test 데이터 없음 (embeddings_ko 비었거나 split 문제)")
        return
    logger.info(f"    test: {Xk.shape}")
    if args.model == "lr":
        pred_ko = predict_lr(MODELS_DIR / "lr_model_ko.pkl", Xk)
    else:
        pred_ko = predict_mlp(MODELS_DIR / "mlp_model_ko.pt", Xk, KO_DIM)
    metrics_ko = compute_metrics(Yk, pred_ko, args.min_support)
    print_metrics(metrics_ko, "한국어 (KcELECTRA)", args.min_support)

    # ── 통합 (영어 + 한국어 합쳐서) ──
    logger.info("\n  [통합] 영어 + 한국어 합산")
    Y_all = np.vstack([Ye, Yk])
    pred_all = np.vstack([pred_en, pred_ko])
    metrics_all = compute_metrics(Y_all, pred_all, args.min_support)
    print_metrics(metrics_all, "전체 (라우팅)", args.min_support)

    # ── 언어별 요약 ──
    logger.info("\n" + "=" * 60)
    logger.info("  언어별 macro-F1 요약")
    logger.info("=" * 60)
    logger.info(f"  영어   (E5):        macro-F1 {metrics_en['macro_f1']:.4f} | micro {metrics_en['micro_f1']:.4f}")
    logger.info(f"  한국어 (KcELECTRA): macro-F1 {metrics_ko['macro_f1']:.4f} | micro {metrics_ko['micro_f1']:.4f}")
    logger.info(f"  전체:               macro-F1 {metrics_all['macro_f1']:.4f} | micro {metrics_all['micro_f1']:.4f}")

    # 저장
    out = {"english": metrics_en, "korean": metrics_ko, "combined": metrics_all,
           "model": args.model}
    with open(RESULTS_DIR / "routed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"\n  저장: {RESULTS_DIR / 'routed_metrics.json'}")
    conn.close()


if __name__ == "__main__":
    main()
