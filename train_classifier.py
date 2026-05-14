"""
train_classifier.py
====================
SQLite의 임베딩 + 라벨로 multi-label 분류기 학습.
2개 모델 비교: Logistic Regression (메인) vs MLP (비교군).

Multi-label 처리:
  - 8개 카테고리 각각에 대해 binary classification
  - LR: OneVsRestClassifier
  - MLP: 마지막 레이어 sigmoid(8) + BCEWithLogitsLoss

저장:
  models/lr_model.pkl
  models/mlp_model.pt

사용:
  python train_classifier.py             # 둘 다 학습
  python train_classifier.py --only lr   # LR만
  python train_classifier.py --only mlp  # MLP만
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH    = Path("data/llm_dataset/dataset.sqlite")
MODELS_DIR = Path("models")
EMBED_DIM  = 384

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_split(conn: sqlite3.Connection, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """split별 (X 임베딩, Y 멀티라벨, ids) 로드"""
    cat_cols = ", ".join(f"l.{c}" for c in CATEGORIES)
    sql = f"""
        SELECT t.id, e.vector, {cat_cols}
        FROM texts t
        JOIN splits s     ON t.id = s.text_id
        JOIN embeddings e ON t.id = e.text_id
        JOIN labels l     ON t.id = l.text_id
        WHERE s.split = ?
    """
    rows = conn.execute(sql, (split,)).fetchall()
    if not rows:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), np.zeros((0, len(CATEGORIES)), dtype=np.int8), []

    ids = [r[0] for r in rows]
    X = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    Y = np.array([r[2:] for r in rows], dtype=np.int8)
    return X, Y, ids


# ──────────────────────────────────────────────
# 모델 1: Logistic Regression (OneVsRest)
# ──────────────────────────────────────────────

def train_lr(X_train, Y_train, X_val, Y_val):
    """OneVsRestClassifier(LogisticRegression) — 카테고리별 binary classifier"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    logger.info(f"  학습 데이터: {X_train.shape}, 라벨: {Y_train.shape}")
    logger.info(f"  카테고리별 양성 비율:")
    for i, c in enumerate(CATEGORIES):
        pos = Y_train[:, i].sum()
        logger.info(f"    {c:20s}: {pos:5d} / {len(Y_train):5d} ({pos/len(Y_train)*100:.1f}%)")

    base = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",   # 불균형 카테고리 (political 등) 보정
        solver="liblinear",
        n_jobs=1,
    )
    model = OneVsRestClassifier(base, n_jobs=-1)

    t0 = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - t0
    logger.info(f"  학습 시간: {train_time:.1f}초")

    # Val에서 빠르게 점검
    Y_pred = model.predict(X_val)
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
    logger.info(f"  [Val] macro-F1: {macro_f1:.4f} | micro-F1: {micro_f1:.4f}")

    return model, {"train_time_sec": train_time, "val_macro_f1": macro_f1, "val_micro_f1": micro_f1}


# ──────────────────────────────────────────────
# 모델 2: MLP (PyTorch)
# ──────────────────────────────────────────────

def train_mlp(X_train, Y_train, X_val, Y_val, epochs: int = 30, lr: float = 1e-3):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  device: {device}")

    # 모델 구조: 384 → 256 → 128 → 8
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

    model = MLP().to(device)

    # 클래스 불균형 보정 — pos_weight
    pos_weights = []
    for i in range(len(CATEGORIES)):
        n_pos = Y_train[:, i].sum()
        n_neg = len(Y_train) - n_pos
        w = (n_neg / max(n_pos, 1))
        pos_weights.append(min(w, 50.0))  # 너무 큰 weight는 cap
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val.astype(np.float32)).to(device)

    train_ds = TensorDataset(Xt, Yt)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    from sklearn.metrics import f1_score
    best_f1 = 0.0
    best_state = None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # Val F1
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_pred = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(np.int8)
        macro_f1 = f1_score(Y_val, val_pred, average="macro", zero_division=0)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  epoch {epoch+1:3d}: loss={epoch_loss:.4f} | val macro-F1={macro_f1:.4f} (best {best_f1:.4f})")

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 최종 val 메트릭
    with torch.no_grad():
        val_logits = model(Xv)
        val_pred = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(np.int8)
    micro_f1 = f1_score(Y_val, val_pred, average="micro", zero_division=0)

    logger.info(f"  학습 시간: {train_time:.1f}초")
    logger.info(f"  [Val] macro-F1: {best_f1:.4f} | micro-F1: {micro_f1:.4f}")

    return model, {"train_time_sec": train_time, "val_macro_f1": best_f1, "val_micro_f1": micro_f1}


# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────

def save_lr(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "categories": CATEGORIES}, f)
    logger.info(f"  저장: {path}")


def save_mlp(model, path: Path):
    import torch
    torch.save({
        "state_dict": model.state_dict(),
        "categories": CATEGORIES,
        "embed_dim": EMBED_DIM,
    }, path)
    logger.info(f"  저장: {path}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lr", "mlp"], default=None,
                    help="특정 모델만 학습 (기본: 둘 다)")
    ap.add_argument("--epochs", type=int, default=30, help="MLP 에폭 수")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("[3단계] 분류기 학습")
    logger.info("=" * 60)

    if not DB_PATH.exists():
        logger.error(f"  DB 없음: {DB_PATH}")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # 데이터 로드
    logger.info("\n  데이터 로드 중...")
    X_train, Y_train, _ = load_split(conn, "train")
    X_val,   Y_val,   _ = load_split(conn, "val")
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    if len(X_train) == 0:
        logger.error("  학습 데이터 없음. embed_texts.py를 먼저 실행했는지 확인.")
        return

    results = {}

    # ── LR ──
    if args.only != "mlp":
        logger.info("\n  ━━━ Logistic Regression (One-vs-Rest) ━━━")
        lr_model, lr_metrics = train_lr(X_train, Y_train, X_val, Y_val)
        save_lr(lr_model, MODELS_DIR / "lr_model.pkl")
        results["LR"] = lr_metrics

    # ── MLP ──
    if args.only != "lr":
        logger.info("\n  ━━━ MLP (PyTorch) ━━━")
        try:
            mlp_model, mlp_metrics = train_mlp(X_train, Y_train, X_val, Y_val, epochs=args.epochs)
            save_mlp(mlp_model, MODELS_DIR / "mlp_model.pt")
            results["MLP"] = mlp_metrics
        except ImportError as e:
            logger.warning(f"  PyTorch 없음 — MLP 건너뜀 ({e})")

    # 비교 출력
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("  모델 비교 (Validation set)")
        logger.info("=" * 60)
        logger.info(f"  {'Model':<8} {'macro-F1':>10} {'micro-F1':>10} {'학습시간':>10}")
        for name, m in results.items():
            logger.info(f"  {name:<8} {m['val_macro_f1']:>10.4f} {m['val_micro_f1']:>10.4f} {m['train_time_sec']:>9.1f}s")
        logger.info("\n  → evaluate.py 로 test set 최종 평가하세요.")

    conn.close()


if __name__ == "__main__":
    main()
