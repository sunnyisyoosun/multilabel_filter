"""
evaluate.py (v2)
================
학습된 분류기를 test set에서 평가.

v2 개선:
  - --min-support: 카테고리별 support가 N 미만이면 macro-F1 계산 시 제외 (소수 카테고리 노이즈 방지)
  - --balanced: pseudo + labeled를 균등하게 섞은 균형 test set으로 평가
  - 한글 폰트 자동 검색/설정 (matplotlib glyph 경고 해결)
  - 통계 진단 정보 강화

사용:
  python evaluate.py                          # 기본 평가
  python evaluate.py --min-support 100        # support<100 카테고리 macro에서 제외
  python evaluate.py --balanced               # pseudo 포함 균형 test set
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
    "profanity", "hate_speech", "gender", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech": "혐오발언", "gender": "성 관련",
    "threat": "살해협박", "political": "정치", "other": "기타유해",
}


def load_test(conn, balanced: bool = False):
    """
    test set 로드.
    
    balanced=False: 기본 test split만 사용 (전체 test)
    balanced=True:  pseudo + labeled를 균등하게 섞어 토픽 균형 맞춤
                    (소수 카테고리에 pseudo의 toxic 라벨 더 포함)
    """
    cat_cols = ", ".join(f"l.{c}" for c in CATEGORIES)

    if balanced:
        # 모든 split 다 가져와서 카테고리별로 균형있게 샘플
        sql = f"""
            SELECT t.id, t.lang, l.label_source, e.vector, {cat_cols}
            FROM texts t
            JOIN embeddings e ON t.id = e.text_id
            JOIN labels l     ON t.id = l.text_id
        """
    else:
        sql = f"""
            SELECT t.id, t.lang, l.label_source, e.vector, {cat_cols}
            FROM texts t
            JOIN splits s     ON t.id = s.text_id
            JOIN embeddings e ON t.id = e.text_id
            JOIN labels l     ON t.id = l.text_id
            WHERE s.split = 'test'
        """
    rows = conn.execute(sql).fetchall()
    if not rows:
        return None

    ids     = [r[0] for r in rows]
    langs   = [r[1] for r in rows]
    sources = [r[2] for r in rows]
    X = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    Y = np.array([r[4:] for r in rows], dtype=np.int8)

    if balanced:
        # 카테고리별 양성 샘플이 너무 적으면 pseudo에서 추가로 끌어옴
        # 단순 전략: pseudo의 양성 모두 + human 정상의 무작위 동수 샘플
        from collections import Counter
        is_pseudo = np.array([s == "llm_pseudo" for s in sources])
        is_human  = ~is_pseudo
        is_toxic  = (Y.sum(axis=1) > 0)

        # 1) pseudo 전부 포함
        # 2) human 정상 중에서 동수 샘플
        n_pseudo = is_pseudo.sum()
        n_human_clean_available = (is_human & ~is_toxic).sum()
        n_human_sample = min(n_pseudo, n_human_clean_available)

        np.random.seed(42)
        pseudo_idx = np.where(is_pseudo)[0]
        human_clean_idx = np.where(is_human & ~is_toxic)[0]
        if n_human_sample < n_human_clean_available:
            human_clean_idx = np.random.choice(human_clean_idx, n_human_sample, replace=False)

        # 토픽 라벨 있는 human 토픽도 포함 (있다면)
        human_toxic_idx = np.where(is_human & is_toxic)[0]

        keep = np.concatenate([pseudo_idx, human_clean_idx, human_toxic_idx])
        np.random.shuffle(keep)
        X = X[keep]; Y = Y[keep]
        ids = [ids[i] for i in keep]; langs = [langs[i] for i in keep]
        sources = [sources[i] for i in keep]

        logger.info(f"  balanced test: total={len(X)}, pseudo={n_pseudo}, "
                    f"human_clean={n_human_sample}, human_toxic={len(human_toxic_idx)}")

    return X, Y, ids, langs, sources


def load_thresholds(model_name, use_tuned):
    """카테고리별 threshold 로드. use_tuned=False면 전부 0.5."""
    if not use_tuned:
        return np.array([0.5] * len(CATEGORIES))
    path = RESULTS_DIR / f"best_thresholds_{model_name}.json"
    if not path.exists():
        logger.warning(f"  threshold 파일 없음: {path} → 0.5 사용")
        return np.array([0.5] * len(CATEGORIES))
    with open(path) as f:
        d = json.load(f)["thresholds"]
    thr = np.array([d.get(c, 0.5) for c in CATEGORIES])
    logger.info(f"  [{model_name}] 튜닝 threshold 적용: " +
                ", ".join(f"{CATEGORY_KO[c]}={d.get(c,0.5):.2f}" for c in CATEGORIES))
    return thr


def predict_lr(X, thresholds=None):
    with open(MODELS_DIR / "lr_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    Y_prob_raw = model.predict_proba(X)
    # OneVsRest predict_proba는 (n, n_cat) 양성확률 반환
    Y_prob = np.asarray(Y_prob_raw)
    if thresholds is None:
        thresholds = np.array([0.5] * len(CATEGORIES))
    Y_pred = np.zeros((len(X), len(CATEGORIES)), dtype=np.int8)
    for i in range(len(CATEGORIES)):
        Y_pred[:, i] = (Y_prob[:, i] >= thresholds[i]).astype(np.int8)
    return Y_pred, Y_prob


def predict_mlp(X, thresholds=None):
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
    if thresholds is None:
        thresholds = np.array([0.5] * len(CATEGORIES))
    Y_pred = np.zeros_like(Y_prob, dtype=np.int8)
    for i in range(len(CATEGORIES)):
        Y_pred[:, i] = (Y_prob[:, i] >= thresholds[i]).astype(np.int8)
    return Y_pred, Y_prob


# ──────────────────────────────────────────────
# 메트릭 계산
# ──────────────────────────────────────────────

def compute_metrics(Y_true, Y_pred, min_support: int = 0):
    """카테고리별 + 평균 메트릭. min_support 미만 카테고리는 별도 macro도 계산."""
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    import numpy as np

    per_cat = []
    f1_per_cat = []
    eligible_idxs = []   # min_support 이상인 카테고리의 인덱스
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
            "eligible":  bool(n_pos_true >= min_support),
        })
        f1_per_cat.append(float(f))
        if n_pos_true >= min_support:
            eligible_idxs.append(i)

    # 표준 메트릭 (전체 카테고리)
    macro_f1    = f1_score(Y_true, Y_pred, average="macro",    zero_division=0)
    micro_f1    = f1_score(Y_true, Y_pred, average="micro",    zero_division=0)
    weighted_f1 = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)

    # 카테고리 필터링한 macro-F1 (소수 카테고리 노이즈 제거용)
    macro_f1_filtered = None
    if min_support > 0 and eligible_idxs:
        macro_f1_filtered = float(np.mean([f1_per_cat[i] for i in eligible_idxs]))

    return {
        "per_category":      per_cat,
        "macro_f1":          float(macro_f1),
        "micro_f1":          float(micro_f1),
        "weighted_f1":       float(weighted_f1),
        "macro_f1_filtered": macro_f1_filtered,
        "min_support":       min_support,
        "eligible_categories": [CATEGORIES[i] for i in eligible_idxs],
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
    min_sup = metrics.get("min_support", 0)
    logger.info(f"\n  ━━━ {title}: 카테고리별 ━━━")
    header = f"  {'카테고리':12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Support':>9s} {'Predicted':>10s}"
    if min_sup > 0:
        header += f"  {'eligible':>8s}"
    logger.info(header)
    for r in metrics["per_category"]:
        line = (f"  {r['category_ko']:12s} {r['precision']:7.3f} {r['recall']:7.3f} "
                f"{r['f1']:7.3f} {r['support']:9d} {r['predicted']:10d}")
        if min_sup > 0:
            line += f"  {'O' if r.get('eligible') else 'x':>8s}"
        logger.info(line)
    logger.info(f"  {'─'*55}")
    logger.info(f"  macro-F1    : {metrics['macro_f1']:.4f}")
    logger.info(f"  micro-F1    : {metrics['micro_f1']:.4f}")
    logger.info(f"  weighted-F1 : {metrics['weighted_f1']:.4f}")
    if metrics.get("macro_f1_filtered") is not None:
        eligible = metrics["eligible_categories"]
        logger.info(f"  macro-F1 (support≥{min_sup}, {len(eligible)}개): {metrics['macro_f1_filtered']:.4f}")


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


_FONT_SETUP_DONE = False


def _setup_korean_font():
    """Matplotlib에 한글 폰트 설정. 시스템에 있는 한글 폰트 자동 검색."""
    global _FONT_SETUP_DONE
    if _FONT_SETUP_DONE:
        return
    _FONT_SETUP_DONE = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 우선순위: 우분투에 흔히 있는 한글 폰트들
        candidates = [
            "NanumGothic", "Nanum Gothic", "Noto Sans CJK KR",
            "Noto Sans KR", "Malgun Gothic", "AppleGothic", "UnDotum",
            "Source Han Sans KR", "DejaVu Sans",
        ]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((c for c in candidates if c in installed), None)
        if chosen:
            plt.rcParams["font.family"] = chosen
            plt.rcParams["axes.unicode_minus"] = False
            logger.info(f"  한글 폰트: {chosen}")
        else:
            logger.warning(f"  한글 폰트 없음. 영문 레이블 사용. "
                           f"`sudo apt install fonts-nanum` 권장")
    except Exception as e:
        logger.warning(f"  폰트 설정 실패: {e}")


def plot_confusion_matrices(Y_true, Y_pred, model_name: str, path: Path):
    """카테고리별 binary confusion matrix를 한 figure에 그림"""
    try:
        _setup_korean_font()
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib 없음 — 시각화 건너뜀")
        return
    from sklearn.metrics import confusion_matrix

    # 한글 폰트 없으면 영문 카테고리명 사용
    import matplotlib
    has_korean = any(k in matplotlib.rcParams["font.family"] for k in
                     ["Nanum", "Noto", "Malgun", "AppleGothic", "UnDotum", "Han"])
    label_fn = (lambda c: CATEGORY_KO[c]) if has_korean else (lambda c: c)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, c in enumerate(CATEGORIES):
        ax = axes[i]
        cm = confusion_matrix(Y_true[:, i], Y_pred[:, i], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(label_fn(c), fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
        ax.set_xlabel("pred"); ax.set_ylabel("true")
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                        color="white" if cm[ii, jj] > cm.max()/2 else "black", fontsize=10)
    # 마지막 1칸 비움 (7카테고리)
    if len(CATEGORIES) < len(axes):
        for j in range(len(CATEGORIES), len(axes)):
            axes[j].axis("off")

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
    ap.add_argument("--min-support", type=int, default=50,
                    help="Macro-F1 계산 시 이 미만 카테고리는 제외 (소수 카테고리 노이즈 방지)")
    ap.add_argument("--use-tuned", action="store_true",
                    help="results/best_thresholds_{model}.json의 카테고리별 threshold 적용")
    ap.add_argument("--balanced", action="store_true",
                    help="Test set에 pseudo도 포함 (영/한 균형, 카테고리 분포 개선)")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("[4단계] 평가")
    logger.info(f"  min_support: {args.min_support}  |  balanced: {args.balanced}")
    logger.info("=" * 60)

    if not DB_PATH.exists():
        logger.error(f"  DB 없음: {DB_PATH}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    test = load_test(conn, balanced=args.balanced)
    if test is None:
        logger.error("  test set이 비어있음.")
        return
    X_test, Y_test, _, langs, sources = test
    logger.info(f"  Test: {X_test.shape}, langs: {set(langs)}, sources: {set(sources)}")

    # 진단: 라벨 출처별 분포
    from collections import Counter
    src_counter = Counter(sources)
    logger.info(f"  test 출처: {dict(src_counter)}")

    all_results = {}
    metrics_lr = None
    metrics_mlp = None

    # ── LR 평가 ──
    if args.only != "mlp" and (MODELS_DIR / "lr_model.pkl").exists():
        logger.info("\n  ━━━ Logistic Regression ━━━")
        thr_lr = load_thresholds("lr", args.use_tuned)
        Y_pred_lr, _ = predict_lr(X_test, thr_lr)
        metrics_lr = compute_metrics(Y_test, Y_pred_lr, min_support=args.min_support)
        per_lang_lr = compute_per_lang_metrics(Y_test, Y_pred_lr, langs)
        print_per_category_table(metrics_lr, "LR")
        print_per_lang(per_lang_lr, "LR")
        all_results["LR"] = {**metrics_lr, "per_language": per_lang_lr}
        plot_confusion_matrices(Y_test, Y_pred_lr, "LR", RESULTS_DIR / "confusion_matrix_lr.png")

    # ── MLP 평가 ──
    if args.only != "lr" and (MODELS_DIR / "mlp_model.pt").exists():
        logger.info("\n  ━━━ MLP ━━━")
        try:
            thr_mlp = load_thresholds("mlp", args.use_tuned)
            Y_pred_mlp, _ = predict_mlp(X_test, thr_mlp)
            metrics_mlp = compute_metrics(Y_test, Y_pred_mlp, min_support=args.min_support)
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
        logger.info("  최종 비교")
        logger.info("=" * 60)
        logger.info(f"  {'Metric':<24s} {'LR':>10s} {'MLP':>10s} {'Diff':>10s}")
        for k in ["macro_f1", "micro_f1", "weighted_f1"]:
            lr_v = metrics_lr[k]; mlp_v = metrics_mlp[k]
            diff = mlp_v - lr_v
            logger.info(f"  {k:<24s} {lr_v:>10.4f} {mlp_v:>10.4f} {diff:>+10.4f}")
        # filtered macro
        lr_f = metrics_lr.get("macro_f1_filtered")
        mlp_f = metrics_mlp.get("macro_f1_filtered")
        if lr_f is not None and mlp_f is not None:
            elig = metrics_lr.get("eligible_categories", [])
            label = f"macro_f1 (support≥{args.min_support})"
            logger.info(f"  {label:<24s} {lr_f:>10.4f} {mlp_f:>10.4f} {mlp_f-lr_f:>+10.4f}")
            logger.info(f"    ↳ 평가 카테고리 ({len(elig)}개): {elig}")
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
