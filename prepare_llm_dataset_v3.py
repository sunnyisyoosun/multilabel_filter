"""
prepare_llm_dataset.py
=======================
LLM Pseudo Labeling용 데이터셋 준비

파이프라인 흐름:

[1] 데이터 로드
    - multilabel_filter.py의 load 함수 재사용
    - 9개 소스 직접 로드

[2] JSON으로 원본 데이터 저장
    - data/llm_dataset/raw/*.json (소스별 저장)

[3] 데이터 품질 관리
    - 결측치 / 중복 / 길이 3↓ 제거

[4] 균형 조정 후 최종 JSON 변환
    - 욕설 : 정상 = 1:1
    - 영어 : 한국어 = 1:1
    - 문장 분리 + 문맥(context) 추가
    - data/llm_dataset/labeled.json
    - data/llm_dataset/pseudo_target.json  (BAD unsafe → LLM 레이블링 대상)
    - data/llm_dataset/stats.json

욕설 타입 (7개):
  profanity / hate_speech / sexual_harassment / sexism / threat
  + political (정치 — attack/hate 통합)
  + other (기타유해 — 외모비하/학력비하/지역비하/나이차별 등)

pseudo_target 비율:
  영어 : 한국어 = 1 : 1
  각 언어 내부에서 유해 추정 : 정상 추정 = 1.3 : 1
  - 영어 유해 = bad_unsafe, 영어 정상 = bad_safe
  - 한국어 유해 = kmhas/unsmile의 is_toxic=1, 한국어 정상 = is_toxic=0
  실제 라벨은 LLM이 결정하므로 위 비율은 입력 분포일 뿐.

⚠ multilabel_filter.py의 CATEGORIES/CATEGORY_KO에도 동일하게 추가 필요.
   각 load_* 함수가 새 카테고리 컬럼을 0으로 채우도록 만들어야 함.
"""

import re
import sys
import json
import logging
import random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from multilabel_filter import (
    CATEGORIES, CATEGORY_KO,
    load_bad, load_hate_speech, load_toxigen,
    load_hatexplain, load_ethos, load_kmhas,
    load_korean_unsmile, load_jigsaw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_aihub(data_dir: str = "/home/ys/engeneer/aihub") -> pd.DataFrame:
    """AIHub 일상대화 — 정상 텍스트 (음성 샘플)"""
    import glob, json as _json
    logger.info("AIHub 일상대화 로드 중...")
    files = []
    for session in ["session2", "session3", "session4"]:
        found = glob.glob(f"{data_dir}/{session}/**/*.txt", recursive=True)
        files.extend(found)
    if not files:
        logger.warning("  AIHub 파일 없음 — 건너뜀")
        return pd.DataFrame(columns=["text", "lang"] + CATEGORIES)
    rows = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = _json.load(f)
            for session in data.get("sessionInfo", []):
                for turn in session.get("dialog", []):
                    utt = turn.get("utterance", "").strip()
                    if utt and len(utt) > 3:
                        rows.append(utt)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["text", "lang"] + CATEGORIES)
    df = pd.DataFrame({"text": rows, "lang": "ko"})
    for cat in CATEGORIES:
        df[cat] = 0  # 전부 정상 샘플
    if len(df) > 30000:
        df = df.sample(30000, random_state=42).reset_index(drop=True)
    logger.info(f"  AIHub 일상대화: {len(df):,}건 (정상 샘플)")
    return df

OUT_DIR     = Path("data/llm_dataset")
RAW_DIR     = OUT_DIR / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def df_to_json(df: pd.DataFrame, path: Path) -> None:
    """DataFrame → JSON 저장"""
    records = df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"  저장: {path.name} ({len(records):,}건)")


def quality_check(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """데이터 품질 관리: 결측치 / 중복 / 길이 3↓ 제거"""
    before = len(df)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 3]
    df = df.drop_duplicates(subset=["text"])
    df[CATEGORIES] = df[CATEGORIES].fillna(0).astype(int)
    after = len(df)
    logger.info(f"  [{label}] 품질 관리: {before:,} → {after:,}건 ({before-after:,}건 제거)")
    return df.reset_index(drop=True)


def balance_1to1(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """욕설:정상 = 1:1 균형"""
    toxic  = df[df[CATEGORIES].sum(axis=1) > 0]
    normal = df[df[CATEGORIES].sum(axis=1) == 0]
    n = min(len(toxic), len(normal))
    result = pd.concat([
        toxic.sample(n, random_state=42),
        normal.sample(n, random_state=42),
    ], ignore_index=True).sample(frac=1, random_state=42)
    logger.info(f"  [{label}] 유해 {n:,} + 정상 {n:,} = {len(result):,}건")
    return result


def split_sentences(text: str) -> list:
    sents = re.split(r'(?<=[.!?])\s*', text.strip())
    return [s.strip() for s in sents if s.strip()] or [text]


def make_records(text: str, lang: str, labels: dict, row_id: str) -> list:
    """문장 분리 + 문맥 포함 레코드 생성"""
    sents = split_sentences(text)
    label_names = [CATEGORY_KO[c] for c in CATEGORIES if labels.get(c, 0) == 1]
    records = []
    for i, sent in enumerate(sents):
        records.append({
            "id":          f"{row_id}_s{i}",
            "full_text":   text,
            "context":     " ".join(sents[:i]),
            "text":        sent,
            "lang":        lang,
            "labels":      {c: labels.get(c, 0) for c in CATEGORIES},
            "label_names": label_names,
            "is_toxic":    len(label_names) > 0,
        })
    return records


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("LLM 데이터셋 준비 파이프라인")
    logger.info("=" * 60)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [1] 데이터 로드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[1/4] 데이터 로드...")
    bad_safe, bad_unsafe = load_bad()
    hate       = load_hate_speech()
    toxigen    = load_toxigen()
    hatexplain = load_hatexplain()
    ethos      = load_ethos()
    jigsaw     = load_jigsaw()
    kmhas      = load_kmhas()
    unsmile    = load_korean_unsmile()
    aihub      = load_aihub()

    # lang 컬럼 추가
    for df, lang in [(hate, "en"), (toxigen, "en"), (hatexplain, "en"),
                     (ethos, "en"), (jigsaw, "en"), (bad_safe, "en"),
                     (bad_unsafe, "en"), (kmhas, "ko"), (unsmile, "ko"), (aihub, "ko")]:
        df["lang"] = lang

    logger.info(f"  소스별 건수:")
    logger.info(f"    영어: hate={len(hate):,} toxigen={len(toxigen):,} hatexplain={len(hatexplain):,} ethos={len(ethos):,} jigsaw={len(jigsaw):,} bad_safe={len(bad_safe):,}")
    logger.info(f"    한국어: kmhas={len(kmhas):,} unsmile={len(unsmile):,} aihub={len(aihub):,}")
    logger.info(f"    pseudo 대상: bad_unsafe={len(bad_unsafe):,}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [2] JSON으로 원본 데이터 저장 (소스별)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[2/4] 소스별 원본 JSON 저장...")
    for name, df in [
        ("hate_speech",  hate),
        ("toxigen",      toxigen),
        ("hatexplain",   hatexplain),
        ("ethos",        ethos),
        ("jigsaw",       jigsaw),
        ("bad_safe",     bad_safe),
        ("kmhas",        kmhas),
        ("unsmile",      unsmile),
        ("aihub",        aihub),
        ("bad_unsafe",   bad_unsafe),
    ]:
        cols = ["text", "lang"] + [c for c in CATEGORIES if c in df.columns]
        df_to_json(df[cols], RAW_DIR / f"{name}.json")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [3] 데이터 품질 관리
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[3/4] 데이터 품질 관리...")
    en_df = pd.concat([hate, toxigen, hatexplain, ethos, jigsaw, bad_safe], ignore_index=True)
    ko_df = pd.concat([kmhas, unsmile, aihub], ignore_index=True)

    en_df = quality_check(en_df, "영어")
    ko_df = quality_check(ko_df, "한국어")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [4] 균형 조정 후 최종 JSON 변환
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logger.info("\n[4/4] 균형 조정 + JSON 변환...")

    # 욕설:정상 = 1:1
    en_bal = balance_1to1(en_df, "영어")
    ko_bal = balance_1to1(ko_df, "한국어")

    # 영어:한국어 = 1:1
    n = min(len(en_bal), len(ko_bal))
    en_final = en_bal.sample(n, random_state=42).reset_index(drop=True)
    ko_final = ko_bal.sample(n, random_state=42).reset_index(drop=True)
    logger.info(f"  영어:한국어 균형 → 각 {n:,}건")

    # 문장 분리 + 문맥 추가 → labeled.json
    labeled_records = []
    for idx, row in en_final.iterrows():
        labels = {c: int(row[c]) for c in CATEGORIES}
        labeled_records.extend(make_records(row["text"], "en", labels, f"en_{idx:06d}"))
    for idx, row in ko_final.iterrows():
        labels = {c: int(row[c]) for c in CATEGORIES}
        labeled_records.extend(make_records(row["text"], "ko", labels, f"ko_{idx:06d}"))
    random.shuffle(labeled_records)

    # pseudo_target.json (LLM 레이블링 대상)
    # 비율 정책:
    #   - 영어 : 한국어 = 1 : 1
    #   - 각 언어 내부에서 유해 추정 : 정상 추정 = 1.3 : 1
    PSEUDO_TOXIC_RATIO = 1.3   # 유해:정상

    # ── 영어: bad_unsafe (유해) + bad_safe (정상) ──
    n_en_unsafe = len(bad_unsafe)
    n_en_safe   = min(int(n_en_unsafe / PSEUDO_TOXIC_RATIO), len(bad_safe))
    en_unsafe = bad_unsafe.reset_index(drop=True)
    en_safe   = (bad_safe.sample(n=n_en_safe, random_state=42).reset_index(drop=True)
                 if n_en_safe < len(bad_safe) else bad_safe.reset_index(drop=True))
    n_en_total = len(en_unsafe) + len(en_safe)

    # ── 한국어: kmhas + unsmile에서 유해/정상 추출 ──
    # is_toxic 플래그는 multilabel_filter에서 카테고리 합으로 만들어졌다고 가정
    ko_pool = pd.concat([kmhas, unsmile], ignore_index=True)
    if "is_toxic" in ko_pool.columns:
        ko_unsafe_pool = ko_pool[ko_pool["is_toxic"] == 1].reset_index(drop=True)
        ko_safe_pool   = ko_pool[ko_pool["is_toxic"] == 0].reset_index(drop=True)
    else:
        # is_toxic 컬럼이 없으면 카테고리 합으로 계산
        toxic_mask = (ko_pool[CATEGORIES].sum(axis=1) > 0)
        ko_unsafe_pool = ko_pool[toxic_mask].reset_index(drop=True)
        ko_safe_pool   = ko_pool[~toxic_mask].reset_index(drop=True)

    # 영:한 1:1 — 한국어 총량을 영어 총량과 맞춤
    n_ko_total  = n_en_total
    n_ko_unsafe = int(n_ko_total * PSEUDO_TOXIC_RATIO / (PSEUDO_TOXIC_RATIO + 1))
    n_ko_safe   = n_ko_total - n_ko_unsafe
    n_ko_unsafe = min(n_ko_unsafe, len(ko_unsafe_pool))
    n_ko_safe   = min(n_ko_safe,   len(ko_safe_pool))

    if n_ko_unsafe < len(ko_unsafe_pool):
        ko_unsafe = ko_unsafe_pool.sample(n=n_ko_unsafe, random_state=42).reset_index(drop=True)
    else:
        ko_unsafe = ko_unsafe_pool
    if n_ko_safe < len(ko_safe_pool):
        ko_safe = ko_safe_pool.sample(n=n_ko_safe, random_state=42).reset_index(drop=True)
    else:
        ko_safe = ko_safe_pool

    logger.info(f"  pseudo_target 구성:")
    logger.info(f"    영어  unsafe={len(en_unsafe):,}, safe={len(en_safe):,}  → 비율 {len(en_unsafe)/max(len(en_safe),1):.2f}:1")
    logger.info(f"    한국어 unsafe={len(ko_unsafe):,}, safe={len(ko_safe):,}  → 비율 {len(ko_unsafe)/max(len(ko_safe),1):.2f}:1")
    logger.info(f"    영:한 = {n_en_total:,} : {len(ko_unsafe)+len(ko_safe):,}")

    # ── pseudo_records 생성 — 라벨 지우고 LLM에 보냄 ──
    pseudo_records = []

    def _add_to_pseudo(df, lang, prefix):
        for idx, row in df.iterrows():
            recs = make_records(row["text"], lang, {c: -1 for c in CATEGORIES}, f"{prefix}_{idx:06d}")
            for r in recs:
                r["labels"]      = {c: None for c in CATEGORIES}
                r["label_names"] = None
                r["is_toxic"]    = None
            pseudo_records.extend(recs)

    _add_to_pseudo(en_unsafe, "en", "bad_unsafe")
    _add_to_pseudo(en_safe,   "en", "bad_safe")
    _add_to_pseudo(ko_unsafe, "ko", "ko_unsafe")
    _add_to_pseudo(ko_safe,   "ko", "ko_safe")

    # 셔플 — LLM 라벨링 시 동일 출처/언어가 몰리지 않도록
    random.shuffle(pseudo_records)

    # 저장
    with open(OUT_DIR / "labeled.json", "w", encoding="utf-8") as f:
        json.dump(labeled_records, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "pseudo_target.json", "w", encoding="utf-8") as f:
        json.dump(pseudo_records, f, ensure_ascii=False, indent=2)

    # 통계
    toxic_cnt  = sum(1 for r in labeled_records if r["is_toxic"])
    normal_cnt = sum(1 for r in labeled_records if not r["is_toxic"])
    stats = {
        "pipeline": {
            "step1": "데이터 로드 (multilabel_filter.py 함수 재사용)",
            "step2": "소스별 JSON 저장 → data/llm_dataset/raw/*.json",
            "step3": "데이터 품질 관리 (결측치/중복/길이 필터링)",
            "step4": "균형 조정 + 문맥 포함 JSON 변환",
        },
        "labeled_total":  len(labeled_records),
        "pseudo_total":   len(pseudo_records),
        "toxic":          toxic_cnt,
        "normal":         normal_cnt,
        "lang_en":        sum(1 for r in labeled_records if r["lang"] == "en"),
        "lang_ko":        sum(1 for r in labeled_records if r["lang"] == "ko"),
        "has_context":    sum(1 for r in labeled_records if r["context"]),
        "categories":     {cat: sum(1 for r in labeled_records if r["labels"].get(cat) == 1) for cat in CATEGORIES},
        "label_types":    len(CATEGORIES),
        "category_names": CATEGORY_KO,
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\n완료!")
    logger.info(f"  labeled.json:       {len(labeled_records):,}건")
    logger.info(f"  pseudo_target.json: {len(pseudo_records):,}건")
    logger.info(f"  욕설/정상: {toxic_cnt:,} / {normal_cnt:,}")
    logger.info(f"  영어/한국어: {stats['lang_en']:,} / {stats['lang_ko']:,}")
    logger.info(f"  문맥 있는 레코드: {stats['has_context']:,}건")
    logger.info(f"\n욕설 타입 ({len(CATEGORIES)}개):")
    for cat in CATEGORIES:
        n_cat = stats["categories"][cat]
        logger.info(f"  {CATEGORY_KO[cat]:10s} ({cat}): {n_cat:,}건")


if __name__ == "__main__":
    main()
