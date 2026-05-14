"""
build_database.py
=================
labeled.json + pseudo_labeled_filtered.jsonl.gz 를 통합해서 SQLite DB로 저장.

스키마:
  texts        — id, text, context, full_text, lang, source
  labels       — text_id, profanity, hate_speech, ..., other (8 카테고리)
                 + label_source ('human' | 'llm_pseudo')
  embeddings   — text_id, vector (BLOB, float32 384차원)
                 (embed_texts.py가 채움)
  splits       — text_id, split ('train' | 'val' | 'test')

사용:
  python build_database.py            # DB 새로 생성
  python build_database.py --rebuild  # 기존 DB 삭제 후 재생성
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 경로
DB_PATH         = Path("data/llm_dataset/dataset.sqlite")
LABELED_PATH    = Path("data/llm_dataset/labeled.json")
PSEUDO_PATH     = Path("data/llm_dataset/pseudo_labeled_filtered.jsonl.gz")

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]

# Train/Val/Test split 비율
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 42


# ──────────────────────────────────────────────
# 스키마
# ──────────────────────────────────────────────

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS texts (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    context     TEXT,
    full_text   TEXT,
    lang        TEXT,
    source      TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    text_id           TEXT PRIMARY KEY,
    {', '.join(f'{c} INTEGER NOT NULL DEFAULT 0' for c in CATEGORIES)},
    is_toxic          INTEGER NOT NULL DEFAULT 0,
    label_source      TEXT NOT NULL,    -- 'human' | 'llm_pseudo'
    toxic_span        TEXT,
    reason            TEXT,
    FOREIGN KEY (text_id) REFERENCES texts(id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    text_id   TEXT PRIMARY KEY,
    model     TEXT NOT NULL,            -- e.g. 'multilingual-e5-small'
    vector    BLOB NOT NULL,            -- float32 raw bytes
    dim       INTEGER NOT NULL,
    FOREIGN KEY (text_id) REFERENCES texts(id)
);

CREATE TABLE IF NOT EXISTS splits (
    text_id   TEXT PRIMARY KEY,
    split     TEXT NOT NULL,            -- 'train' | 'val' | 'test'
    FOREIGN KEY (text_id) REFERENCES texts(id)
);

CREATE INDEX IF NOT EXISTS idx_labels_source ON labels(label_source);
CREATE INDEX IF NOT EXISTS idx_labels_toxic  ON labels(is_toxic);
CREATE INDEX IF NOT EXISTS idx_splits_split  ON splits(split);
CREATE INDEX IF NOT EXISTS idx_texts_lang    ON texts(lang);
"""


# ──────────────────────────────────────────────
# DB 초기화
# ──────────────────────────────────────────────

def init_db(db_path: Path, rebuild: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and db_path.exists():
        db_path.unlink()
        logger.info(f"  기존 DB 삭제: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_labeled() -> list[dict]:
    if not LABELED_PATH.exists():
        logger.warning(f"  {LABELED_PATH} 없음 — human 라벨 데이터 건너뜀")
        return []
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"  labeled.json: {len(records):,}건")
    return records


def load_pseudo() -> list[dict]:
    if not PSEUDO_PATH.exists():
        logger.warning(f"  {PSEUDO_PATH} 없음 — pseudo 라벨 건너뜀")
        return []
    records = []
    with gzip.open(PSEUDO_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    logger.info(f"  pseudo_labeled_filtered.jsonl.gz: {len(records):,}건")
    return records


# ──────────────────────────────────────────────
# DB 삽입
# ──────────────────────────────────────────────

def insert_records(conn: sqlite3.Connection, records: list[dict], label_source: str):
    """labeled (human) + pseudo (llm) 양쪽 모두 처리"""
    cur = conn.cursor()
    n_inserted = 0
    n_skipped  = 0

    cat_cols = ", ".join(CATEGORIES)
    cat_placeholders = ", ".join("?" * len(CATEGORIES))

    for r in records:
        text_id = r.get("id")
        text    = r.get("text", "")
        if not text_id or not text:
            n_skipped += 1
            continue

        # texts 테이블 — INSERT OR IGNORE (이미 있으면 패스)
        cur.execute(
            "INSERT OR IGNORE INTO texts (id, text, context, full_text, lang, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                text_id,
                text,
                r.get("context", ""),
                r.get("full_text", ""),
                r.get("lang", ""),
                r.get("source", label_source),
            )
        )

        # labels 테이블 — labeled.json은 카테고리 컬럼이 직접, pseudo는 nested dict
        labels_dict = r.get("labels", {})
        cat_values = [int(labels_dict.get(c, 0)) for c in CATEGORIES]
        is_toxic = 1 if any(cat_values) else 0

        cur.execute(
            f"INSERT OR REPLACE INTO labels "
            f"(text_id, {cat_cols}, is_toxic, label_source, toxic_span, reason) "
            f"VALUES (?, {cat_placeholders}, ?, ?, ?, ?)",
            (text_id, *cat_values, is_toxic, label_source,
             r.get("toxic_span", ""), r.get("reason", ""))
        )
        n_inserted += 1

    conn.commit()
    logger.info(f"  [{label_source}] insert {n_inserted:,}건 (skip {n_skipped:,}건)")


# ──────────────────────────────────────────────
# Train/Val/Test split (Stratified by lang + is_toxic)
# ──────────────────────────────────────────────

def make_splits(conn: sqlite3.Connection):
    """언어 + toxic 여부로 stratify해서 split 생성. 재현 가능 (seed=42)."""
    cur = conn.cursor()
    rng = random.Random(SEED)

    # 기존 splits 삭제 후 재생성
    cur.execute("DELETE FROM splits")

    # 그룹별로 stratify
    cur.execute("""
        SELECT t.id, t.lang, l.is_toxic
        FROM texts t LEFT JOIN labels l ON t.id = l.text_id
    """)
    rows = cur.fetchall()

    # 그룹화: (lang, is_toxic) → ids
    groups = {}
    for tid, lang, is_toxic in rows:
        key = (lang or "?", is_toxic or 0)
        groups.setdefault(key, []).append(tid)

    train, val, test = SPLIT_RATIOS
    n_train, n_val, n_test = 0, 0, 0
    for key, ids in groups.items():
        rng.shuffle(ids)
        n = len(ids)
        i_train = int(n * train)
        i_val   = i_train + int(n * val)
        for tid in ids[:i_train]:
            cur.execute("INSERT INTO splits VALUES (?, ?)", (tid, "train"))
            n_train += 1
        for tid in ids[i_train:i_val]:
            cur.execute("INSERT INTO splits VALUES (?, ?)", (tid, "val"))
            n_val += 1
        for tid in ids[i_val:]:
            cur.execute("INSERT INTO splits VALUES (?, ?)", (tid, "test"))
            n_test += 1

    conn.commit()
    logger.info(f"  splits: train {n_train:,} / val {n_val:,} / test {n_test:,}")


# ──────────────────────────────────────────────
# 통계 출력
# ──────────────────────────────────────────────

def print_stats(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM texts")
    total = cur.fetchone()[0]

    cur.execute("SELECT label_source, COUNT(*) FROM labels GROUP BY label_source")
    by_source = dict(cur.fetchall())

    cur.execute("SELECT lang, COUNT(*) FROM texts GROUP BY lang ORDER BY COUNT(*) DESC")
    by_lang = cur.fetchall()

    cur.execute("SELECT is_toxic, COUNT(*) FROM labels GROUP BY is_toxic")
    by_toxic = dict(cur.fetchall())

    logger.info(f"\n  === DB 통계 ===")
    logger.info(f"  총 텍스트: {total:,}건")
    logger.info(f"  라벨 출처: {by_source}")
    logger.info(f"  언어별: {dict(by_lang[:5])}{'...' if len(by_lang)>5 else ''}")
    logger.info(f"  유해/정상: {by_toxic.get(1,0):,} / {by_toxic.get(0,0):,}")

    logger.info(f"  카테고리별:")
    for c in CATEGORIES:
        cur.execute(f"SELECT COUNT(*) FROM labels WHERE {c} = 1")
        cnt = cur.fetchone()[0]
        logger.info(f"    {c:20s}: {cnt:,}건")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="기존 DB 삭제 후 재생성")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("[1단계] SQLite DB 구축")
    logger.info("=" * 60)

    conn = init_db(DB_PATH, rebuild=args.rebuild)

    # 1) Human 라벨 (labeled.json)
    labeled = load_labeled()
    if labeled:
        insert_records(conn, labeled, label_source="human")

    # 2) LLM Pseudo 라벨 (pseudo_labeled_filtered.jsonl.gz)
    pseudo = load_pseudo()
    if pseudo:
        insert_records(conn, pseudo, label_source="llm_pseudo")

    # 3) Train/Val/Test split
    make_splits(conn)

    # 4) 통계
    print_stats(conn)

    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"\n  DB 파일: {DB_PATH} ({db_size_mb:.2f} MB)")
    conn.close()


if __name__ == "__main__":
    main()
