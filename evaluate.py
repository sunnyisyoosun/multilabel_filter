"""
evaluate.py
============
학습된 분류기를 test set에서 평가.
발표용 결과물 생성:
  - 카테고리별 Precision / Recall / F1 표
  - macro / micro / weighted F1 비교
  - per-language 성능 (영/한)
  - confusion matrix (카테고리별 binary)
  - LR vs MLP 비교 표

출력:
  results/metrics.json
  results/comparison.csv
  results/confusion_matrix.png  (matplotlib 있으면)
  results/per_category.csv

사용:
  python evaluate.py
  python evaluate.py --only lr
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

DB_PATH      = Path("data/llm_dataset/dataset.sqlite")
MODELS_DIR   = Path("models")
RESULTS_DIR  = Path("results")
EMBED_DIM    = 384

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech":       "혐오발언", "sexual_harassment": "성희롱",
    "sexism": "성차별", "threat": "살해협박", "political": "정치",
    "other": "기타유해",
}


def load_test(conn):
    cat_cols = ", ".join(f"l.{c}" for c in CATEGORIES)
    sql = f"""
        SELECT t.id, t.lang, e.vector, {cat_cols}
        FROM texts t
        JOIN splits s     ON t.id = s.text_id
        JOIN embeddings e ON t.id = e.text_id
        JOIN labels l     ON t.id = l.text_id
        WHERE s.split = 'test'
    """
    rows = conn.execute(sql).fetchall()
    if not rows:
        return None
    ids   = [r[0] for r in rows]
    langs = [r[1] for r in rows]
    X = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    Y = np.array([r[3:] for r in rows], dtype=np.int8)
    return X, Y, ids, langs


def predict_lr(X):
    with open(MODELS_DIR / "lr_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    Y_pred = model.predict(X).astype(np.int8)
    Y_prob = model.predict_proba(X)
    return Y_pred, Y_prob


def predict_mlp(X):
    import torch
    import torch.nn as nn
    bundle = torch.load(MODELS_DIR / "mlp_model.pt", weights_only=False)

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

    model = MLP()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        Y_prob = torch.sigmoid(logits).numpy()
    Y_pred = (Y_prob > 0.5).astype(np.int8)
    return Y_pred, Y_prob


# ──────────────────────────────────────────────
# 메트릭 계산
# ──────────────────────────────────────────────

def compute_metrics(Y_true, Y_pred):
    """카테고리별 + 평균 메트릭"""
    from sklearn.metrics import precision_recall_fscore_support, f1_score

    per_cat = []
    for i, c in enumerate(CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            Y_true[:, i], Y_pred[:, i],
            average="binary", zero_division=0,
        )
        n_pos_true = int(Y_true[:, i].sum())
        n_pos_pred = int(Y_pred[:, i].sum())
        per_cat.append({
            "category": c,
            "category_ko": CATEGORY_KO[c],
            "precision": float(p),
            "recall":    float(r),
            "f1":        float(f),
            "support":   n_pos_true,
            "predicted": n_pos_pred,
        })

    macro_f1    = f1_score(Y_true, Y_pred, average="macro",    zero_division=0)
    micro_f1    = f1_score(Y_true, Y_pred, average="micro",    zero_division=0)
    weighted_f1 = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)

    return {
        "per_category": per_cat,
        "macro_f1":     float(macro_f1),
        "micro_f1":     float(micro_f1),
        "weighted_f1":  float(weighted_f1),
    }


def compute_per_lang_metrics(Y_true, Y_pred, langs):
    """언어별 macro-F1"""
    from sklearn.metrics import f1_score
    out = {}
    for lang in set(langs):
        mask = np.array([l == lang for l in langs])
        if mask.sum() < 5:
            continue
        out[lang] = {
            "n":        int(mask.sum()),
            "macro_f1": float(f1_score(Y_true[mask], Y_pred[mask], average="macro", zero_division=0)),
            "micro_f1": float(f1_score(Y_true[mask], Y_pred[mask], average="micro", zero_division=0)),
        }
    return out


# ──────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────

def print_per_category_table(metrics, title):
    logger.info(f"\n  ━━━ {title}: 카테고리별 ━━━")
    logger.info(f"  {'카테고리':12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Support':>9s} {'Predicted':>10s}")
    for r in metrics["per_category"]:
        logger.info(
            f"  {r['category_ko']:12s} {r['precision']:7.3f} {r['recall']:7.3f} "
            f"{r['f1']:7.3f} {r['support']:9d} {r['predicted']:10d}"
        )
    logger.info(f"  {'─'*55}")
    logger.info(f"  macro-F1   : {metrics['macro_f1']:.4f}")
    logger.info(f"  micro-F1   : {metrics['micro_f1']:.4f}")
    logger.info(f"  weighted-F1: {metrics['weighted_f1']:.4f}")


def print_per_lang(per_lang, title):
    if not per_lang:
        return
    logger.info(f"\n  ━━━ {title}: 언어별 ━━━")
    logger.info(f"  {'lang':6s} {'n':>7s} {'macro-F1':>10s} {'micro-F1':>10s}")
    for lang, m in sorted(per_lang.items()):
        logger.info(f"  {lang:6s} {m['n']:>7d} {m['macro_f1']:>10.4f} {m['micro_f1']:>10.4f}")


def save_csv(per_cat_lr, per_cat_mlp, path: Path):
    """카테고리별 비교 CSV 저장"""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if per_cat_mlp:
            w.writerow(["category", "category_ko",
                        "lr_precision", "lr_recall", "lr_f1",
                        "mlp_precision", "mlp_recall", "mlp_f1",
                        "support"])
            for lr, mlp in zip(per_cat_lr, per_cat_mlp):
                w.writerow([lr["category"], lr["category_ko"],
                           f"{lr['precision']:.4f}", f"{lr['recall']:.4f}", f"{lr['f1']:.4f}",
                           f"{mlp['precision']:.4f}", f"{mlp['recall']:.4f}", f"{mlp['f1']:.4f}",
                           lr["support"]])
        else:
            w.writerow(["category", "category_ko", "precision", "recall", "f1", "support"])
            for r in per_cat_lr:
                w.writerow([r["category"], r["category_ko"],
                           f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                           r["support"]])
    logger.info(f"  저장: {path}")


def plot_confusion_matrices(Y_true, Y_pred, model_name: str, path: Path):
    """카테고리별 binary confusion matrix를 한 figure에 그림"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib 없음 — 시각화 건너뜀")
        return
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, c in enumerate(CATEGORIES):
        ax = axes[i]
        cm = confusion_matrix(Y_true[:, i], Y_pred[:, i], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(f"{CATEGORY_KO[c]}", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
        ax.set_xlabel("pred"); ax.set_ylabel("true")
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                        color="white" if cm[ii, jj] > cm.max()/2 else "black", fontsize=10)
    plt.suptitle(f"Confusion Matrices ({model_name})")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  저장: {path}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lr", "mlp"], default=None)
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("[4단계] 평가 (Test set)")
    logger.info("=" * 60)

    if not DB_PATH.exists():
        logger.error(f"  DB 없음: {DB_PATH}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    test = load_test(conn)
    if test is None:
        logger.error("  test set이 비어있음.")
        return
    X_test, Y_test, _, langs = test
    logger.info(f"  Test: {X_test.shape}, langs: {set(langs)}")

    all_results = {}
    metrics_lr = None
    metrics_mlp = None

    # ── LR 평가 ──
    if args.only != "mlp" and (MODELS_DIR / "lr_model.pkl").exists():
        logger.info("\n  ━━━ Logistic Regression ━━━")
        Y_pred_lr, _ = predict_lr(X_test)
        metrics_lr = compute_metrics(Y_test, Y_pred_lr)
        per_lang_lr = compute_per_lang_metrics(Y_test, Y_pred_lr, langs)
        print_per_category_table(metrics_lr, "LR")
        print_per_lang(per_lang_lr, "LR")
        all_results["LR"] = {**metrics_lr, "per_language": per_lang_lr}
        plot_confusion_matrices(Y_test, Y_pred_lr, "LR", RESULTS_DIR / "confusion_matrix_lr.png")

    # ── MLP 평가 ──
    if args.only != "lr" and (MODELS_DIR / "mlp_model.pt").exists():
        logger.info("\n  ━━━ MLP ━━━")
        try:
            Y_pred_mlp, _ = predict_mlp(X_test)
            metrics_mlp = compute_metrics(Y_test, Y_pred_mlp)
            per_lang_mlp = compute_per_lang_metrics(Y_test, Y_pred_mlp, langs)
            print_per_category_table(metrics_mlp, "MLP")
            print_per_lang(per_lang_mlp, "MLP")
            all_results["MLP"] = {**metrics_mlp, "per_language": per_lang_mlp}
            plot_confusion_matrices(Y_test, Y_pred_mlp, "MLP", RESULTS_DIR / "confusion_matrix_mlp.png")
        except ImportError as e:
            logger.warning(f"  PyTorch 없음 — MLP 건너뜀 ({e})")

    # ── 비교 ──
    if metrics_lr and metrics_mlp:
        logger.info("\n" + "=" * 60)
        logger.info("  최종 비교 (Test set)")
        logger.info("=" * 60)
        logger.info(f"  {'Metric':<15s} {'LR':>10s} {'MLP':>10s} {'Diff':>10s}")
        for k in ["macro_f1", "micro_f1", "weighted_f1"]:
            lr_v = metrics_lr[k]; mlp_v = metrics_mlp[k]
            diff = mlp_v - lr_v
            logger.info(f"  {k:<15s} {lr_v:>10.4f} {mlp_v:>10.4f} {diff:>+10.4f}")
        save_csv(metrics_lr["per_category"], metrics_mlp["per_category"], RESULTS_DIR / "comparison.csv")
    elif metrics_lr:
        save_csv(metrics_lr["per_category"], None, RESULTS_DIR / "per_category_lr.csv")
    elif metrics_mlp:
        save_csv(metrics_mlp["per_category"], None, RESULTS_DIR / "per_category_mlp.csv")

    # JSON 덤프
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n  저장: {RESULTS_DIR/'metrics.json'}")
    conn.close()


if __name__ == "__main__":
    main()
