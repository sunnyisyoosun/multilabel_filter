"""
embed_texts.py
==============
SQLite DB의 텍스트들을 multilingual-e5-small로 임베딩.
임베딩 결과는 DB의 embeddings 테이블에 BLOB(float32 384차원)으로 저장.

E5 모델 사용 규칙:
  - 입력 텍스트 앞에 "query: " 또는 "passage: " 접두사 필수
  - 분류 작업에서는 "query: " 사용

설치:
  pip install sentence-transformers torch numpy

사용:
  python embed_texts.py              # 임베딩 안 된 것만 추가
  python embed_texts.py --batch 64   # 배치 크기 조정
  python embed_texts.py --device cuda  # GPU 사용 (있으면)
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
MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_DIM  = 384  # E5-small 차원

E5_PREFIX = "query: "  # 분류 작업용. retrieval-passage라면 "passage: "


def get_pending_texts(conn: sqlite3.Connection, limit: int | None = None) -> list[tuple[str, str]]:
    """임베딩 안 된 텍스트만 가져옴 (재시작 가능)"""
    sql = """
        SELECT t.id, t.text
        FROM texts t
        LEFT JOIN embeddings e ON t.id = e.text_id
        WHERE e.text_id IS NULL
    """
    if limit:
        sql += f" LIMIT {limit}"
    return conn.execute(sql).fetchall()


def encode_batch(model, texts: list[str], batch_size: int) -> np.ndarray:
    """E5 prefix 적용해서 배치 인코딩. L2-normalized 반환 (cosine similarity 용도)."""
    prefixed = [E5_PREFIX + (t or "")[:512] for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2 norm — 분류기에 더 안정적
        convert_to_numpy=True,
    ).astype(np.float32)
    return embs


def insert_embeddings(conn: sqlite3.Connection, ids: list[str], embs: np.ndarray):
    """배치 단위로 INSERT"""
    cur = conn.cursor()
    rows = [
        (tid, MODEL_NAME, emb.tobytes(), EMBED_DIM)
        for tid, emb in zip(ids, embs)
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO embeddings (text_id, model, vector, dim) VALUES (?, ?, ?, ?)",
        rows
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32, help="배치 크기 (CPU 32, GPU 128~256 추천)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--limit", type=int, default=None, help="처리 개수 제한 (테스트용)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        logger.error(f"  DB가 없습니다: {DB_PATH}")
        logger.error(f"  먼저 build_database.py를 실행하세요.")
        return

    logger.info("=" * 60)
    logger.info("[2단계] Multilingual E5-small 임베딩")
    logger.info("=" * 60)

    # device 결정
    device = args.device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    logger.info(f"  device: {device}, batch: {args.batch}")

    # 모델 로드
    from sentence_transformers import SentenceTransformer
    logger.info(f"  모델 로드 중: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info(f"  로드 완료 ({time.time()-t0:.1f}s)")

    # DB 연결
    conn = sqlite3.connect(DB_PATH)

    # 처리 대상
    pending = get_pending_texts(conn, limit=args.limit)
    if not pending:
        logger.info("  모든 텍스트가 이미 임베딩됨.")
        conn.close()
        return
    logger.info(f"  처리할 텍스트: {len(pending):,}건")

    # 배치 처리
    t_start = time.time()
    n_done = 0
    BUFFER_BATCHES = 10  # 10 배치마다 DB flush

    pbar = tqdm(total=len(pending), desc="Embedding")
    buffer_ids: list[str] = []
    buffer_texts: list[str] = []

    for tid, txt in pending:
        buffer_ids.append(tid)
        buffer_texts.append(txt)

        if len(buffer_ids) >= args.batch * BUFFER_BATCHES:
            embs = encode_batch(model, buffer_texts, args.batch)
            insert_embeddings(conn, buffer_ids, embs)
            n_done += len(buffer_ids)
            pbar.update(len(buffer_ids))
            buffer_ids.clear(); buffer_texts.clear()

            elapsed = time.time() - t_start
            speed = n_done / elapsed
            remain = (len(pending) - n_done) / max(speed, 0.01) / 60
            pbar.set_postfix(speed=f"{speed:.0f}/s", eta=f"{remain:.1f}min")

    # 남은 버퍼
    if buffer_ids:
        embs = encode_batch(model, buffer_texts, args.batch)
        insert_embeddings(conn, buffer_ids, embs)
        n_done += len(buffer_ids)
        pbar.update(len(buffer_ids))
    pbar.close()

    elapsed = time.time() - t_start
    logger.info(f"\n  완료: {n_done:,}건 ({elapsed/60:.1f}분, {n_done/max(elapsed,0.01):.0f}건/초)")

    # 통계
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM embeddings")
    total_embs = cur.fetchone()[0]
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"  DB의 총 임베딩: {total_embs:,}건")
    logger.info(f"  DB 파일 크기: {db_size_mb:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
