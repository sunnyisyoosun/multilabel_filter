"""
embed_texts_ko.py
==================
한국어 텍스트만 KcELECTRA로 재임베딩.
mean pooling으로 768-dim 벡터 추출 → embeddings_ko 테이블에 저장.

KcELECTRA-base-v2022 (Beomi):
  - 댓글 데이터로 pretraining (욕설/혐오 도메인 친화)
  - Korean Hate Speech 벤치마크 strong
  - 768-dim 출력

설치:
  pip install transformers torch tqdm

사용:
  python embed_texts_ko.py              # 한국어만 임베딩
  python embed_texts_ko.py --batch 32   # 배치 크기 조정
  python embed_texts_ko.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH    = Path("data/llm_dataset/dataset.sqlite")
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
EMBED_DIM  = 768  # KcELECTRA 출력 차원
MAX_LEN    = 128


# ──────────────────────────────────────────────
# DB 스키마 - embeddings_ko 테이블 추가
# ──────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embeddings_ko (
    text_id   TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    vector    BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    FOREIGN KEY (text_id) REFERENCES texts(id)
);
"""


def init_ko_table(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def get_pending_ko_texts(conn: sqlite3.Connection, limit: int | None = None) -> list[tuple[str, str]]:
    """임베딩 안 된 한국어 텍스트만 가져옴"""
    sql = """
        SELECT t.id, t.text
        FROM texts t
        LEFT JOIN embeddings_ko e ON t.id = e.text_id
        WHERE t.lang = 'ko' AND e.text_id IS NULL
    """
    if limit:
        sql += f" LIMIT {limit}"
    return conn.execute(sql).fetchall()


# ──────────────────────────────────────────────
# KcELECTRA 임베딩 추출
# ──────────────────────────────────────────────

def encode_batch(model, tokenizer, texts: list[str], device: str, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Mean pooling을 사용한 sentence embedding 추출.
    
    1. 토큰화 (padding=True, truncation=True)
    2. KcELECTRA forward
    3. last_hidden_state의 mean pooling (attention_mask 고려)
    4. L2 normalize
    """
    import torch
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**enc)

    # last_hidden_state: (batch, seq_len, hidden)
    hidden = out.last_hidden_state
    mask   = enc["attention_mask"].unsqueeze(-1).float()
    # masked mean
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts

    # L2 normalize
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    return pooled.cpu().numpy().astype(np.float32)


def insert_embeddings(conn: sqlite3.Connection, ids: list[str], embs: np.ndarray):
    cur = conn.cursor()
    rows = [
        (tid, MODEL_NAME, emb.tobytes(), EMBED_DIM)
        for tid, emb in zip(ids, embs)
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO embeddings_ko (text_id, model, vector, dim) VALUES (?, ?, ?, ?)",
        rows
    )
    conn.commit()


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16, help="배치 크기 (CPU: 8~16, GPU: 32~64)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--limit", type=int, default=None, help="처리 개수 제한 (테스트용)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        logger.error(f"  DB 없음: {DB_PATH}")
        return

    logger.info("=" * 60)
    logger.info("[추가] KcELECTRA 한국어 임베딩")
    logger.info("=" * 60)

    # device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info(f"  device: {device}, batch: {args.batch}")

    # 모델 로드
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        logger.error("  transformers/torch 설치 필요: pip install transformers torch")
        return

    logger.info(f"  모델 로드 중: {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    logger.info(f"  로드 완료 ({time.time()-t0:.1f}s)")

    # DB 연결 + 테이블 초기화
    conn = sqlite3.connect(DB_PATH)
    init_ko_table(conn)

    # 처리 대상
    pending = get_pending_ko_texts(conn, limit=args.limit)
    if not pending:
        logger.info("  모든 한국어 텍스트 이미 임베딩됨.")
        conn.close()
        return
    logger.info(f"  처리할 한국어 텍스트: {len(pending):,}건")

    # 배치 처리
    t_start = time.time()
    n_done = 0
    BUFFER_BATCHES = 8

    pbar = tqdm(total=len(pending), desc="KcELECTRA")
    buffer_ids: list[str] = []
    buffer_texts: list[str] = []

    for tid, txt in pending:
        buffer_ids.append(tid)
        buffer_texts.append(txt[:512])  # truncation

        if len(buffer_ids) >= args.batch * BUFFER_BATCHES:
            embs = encode_batch(model, tokenizer, buffer_texts, device)
            insert_embeddings(conn, buffer_ids, embs)
            n_done += len(buffer_ids)
            pbar.update(len(buffer_ids))
            buffer_ids.clear(); buffer_texts.clear()

            elapsed = time.time() - t_start
            speed = n_done / elapsed
            remain = (len(pending) - n_done) / max(speed, 0.01) / 60
            pbar.set_postfix(speed=f"{speed:.0f}/s", eta=f"{remain:.1f}min")

    if buffer_ids:
        embs = encode_batch(model, tokenizer, buffer_texts, device)
        insert_embeddings(conn, buffer_ids, embs)
        n_done += len(buffer_ids)
        pbar.update(len(buffer_ids))
    pbar.close()

    elapsed = time.time() - t_start
    logger.info(f"\n  완료: {n_done:,}건 ({elapsed/60:.1f}분, {n_done/max(elapsed,0.01):.0f}건/초)")

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM embeddings_ko")
    total = cur.fetchone()[0]
    logger.info(f"  총 KcELECTRA 임베딩: {total:,}건")
    conn.close()


if __name__ == "__main__":
    main()
