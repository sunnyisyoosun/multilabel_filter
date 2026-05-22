"""
train_classifier_ko.py
=======================
KcELECTRA 임베딩(embeddings_ko 테이블) 기반 한국어 전용 분류기 학습.

설계: train_classifier.py와 동일한 구조, embeddings_ko 사용 + lang='ko' 필터링.
출력: models/lr_model_ko.pkl, models/mlp_model_ko.pt

사용:
  python train_classifier_ko.py
  python train_classifier_ko.py --only lr
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH    = Path("data/llm_dataset/dataset.sqlite")
MODELS_DIR = Path("models")
EMBED_DIM  = 768   # KcELECTRA

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]


def load_split_ko(conn: sqlite3.Connection, split: str):
    """한국어만 + embeddings_ko 사용"""
    cat_cols = ", ".join(f"l.{c}" for c in CATEGORIES)
    sql = f"""
        SELECT t.id, e.vector, {cat_cols}
        FROM texts t
        JOIN splits s        ON t.id = s.text_id
        JOIN embeddings_ko e ON t.id = e.text_id
        JOIN labels l        ON t.id = l.text_id
        WHERE s.split = ? AND t.lang = 'ko'
    """
    rows = conn.execute(sql, (split,)).fetchall()
    if not rows:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), np.zeros((0, len(CATEGORIES)), dtype=np.int8), []
    ids = [r[0] for r in rows]
    X = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    Y = np.array([r[2:] for r in rows], dtype=np.int8)
    return X, Y, ids


def train_lr_ko(X_train, Y_train, X_val, Y_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import f1_score

    logger.info(f"  학습 데이터: {X_train.shape}")

    base = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", solver="liblinear", n_jobs=1,
    )
    model = OneVsRestClassifier(base, n_jobs=-1)

    t0 = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - t0
    logger.info(f"  학습 시간: {train_time:.1f}초")

    Y_pred = model.predict(X_val)
    macro_f1 = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
    logger.info(f"  [Val-KO] macro-F1: {macro_f1:.4f} | micro-F1: {micro_f1:.4f}")
    return model, {"train_time_sec": train_time, "val_macro_f1": macro_f1, "val_micro_f1": micro_f1}


def train_mlp_ko(X_train, Y_train, X_val, Y_val, epochs: int = 30, lr: float = 1e-3):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  device: {device}")

    # MLP: 768 → 256 → 128 → 7
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

    # pos_weight (불균형 보정)
    pos_weights = []
    for i in range(len(CATEGORIES)):
        n_pos = Y_train[:, i].sum()
        n_neg = len(Y_train) - n_pos
        w = (n_neg / max(n_pos, 1))
        pos_weights.append(min(w, 50.0))
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_val).to(device)

    train_ds = TensorDataset(Xt, Yt)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

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

    with torch.no_grad():
        val_logits = model(Xv)
        val_pred = (torch.sigmoid(val_logits) > 0.5).cpu().numpy().astype(np.int8)
    micro_f1 = f1_score(Y_val, val_pred, average="micro", zero_division=0)

    logger.info(f"  학습 시간: {train_time:.1f}초")
    logger.info(f"  [Val-KO] macro-F1: {best_f1:.4f} | micro-F1: {micro_f1:.4f}")
    return model, {"train_time_sec": train_time, "val_macro_f1": best_f1, "val_micro_f1": micro_f1}


def save_lr(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "categories": CATEGORIES, "embed_dim": EMBED_DIM,
                     "embedder": "KcELECTRA"}, f)
    logger.info(f"  저장: {path}")


def save_mlp(model, path: Path):
    import torch
    torch.save({
        "state_dict": model.state_dict(),
        "categories": CATEGORIES,
        "embed_dim":  EMBED_DIM,
        "embedder":   "KcELECTRA",
    }, path)
    logger.info(f"  저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lr", "mlp"], default=None)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("[KO] 한국어 분류기 학습 (KcELECTRA)")
    logger.info("=" * 60)

    if not DB_PATH.exists():
        logger.error(f"  DB 없음: {DB_PATH}")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # 한국어 임베딩 있는지 확인
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM embeddings_ko")
    n_emb = cur.fetchone()[0]
    if n_emb == 0:
        logger.error("  embeddings_ko 테이블이 비어있음. embed_texts_ko.py 먼저 실행.")
        return
    logger.info(f"  KcELECTRA 임베딩: {n_emb:,}건")

    logger.info("\n  데이터 로드 중...")
    X_train, Y_train, _ = load_split_ko(conn, "train")
    X_val,   Y_val,   _ = load_split_ko(conn, "val")
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    if len(X_train) == 0:
        logger.error("  학습 데이터 없음.")
        return

    # 카테고리별 양성 비율
    logger.info(f"\n  카테고리별 양성 비율 (train):")
    for i, c in enumerate(CATEGORIES):
        pos = Y_train[:, i].sum()
        logger.info(f"    {c:20s}: {pos:5d} / {len(Y_train):5d} ({pos/max(len(Y_train),1)*100:.1f}%)")

    results = {}

    if args.only != "mlp":
        logger.info("\n  ━━━ LR (KcELECTRA + OneVsRest) ━━━")
        lr_model, lr_metrics = train_lr_ko(X_train, Y_train, X_val, Y_val)
        save_lr(lr_model, MODELS_DIR / "lr_model_ko.pkl")
        results["LR"] = lr_metrics

    if args.only != "lr":
        logger.info("\n  ━━━ MLP (KcELECTRA + PyTorch) ━━━")
        try:
            mlp_model, mlp_metrics = train_mlp_ko(X_train, Y_train, X_val, Y_val, epochs=args.epochs)
            save_mlp(mlp_model, MODELS_DIR / "mlp_model_ko.pt")
            results["MLP"] = mlp_metrics
        except ImportError as e:
            logger.warning(f"  PyTorch 없음 — MLP 건너뜀 ({e})")

    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("  한국어 모델 비교 (Validation)")
        logger.info("=" * 60)
        logger.info(f"  {'Model':<8} {'macro-F1':>10} {'micro-F1':>10} {'학습시간':>10}")
        for name, m in results.items():
            logger.info(f"  {name:<8} {m['val_macro_f1']:>10.4f} {m['val_micro_f1']:>10.4f} {m['train_time_sec']:>9.1f}s")

    conn.close()


if __name__ == "__main__":
    main()
