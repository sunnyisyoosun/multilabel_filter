"""
filter_pseudo_labels.py
========================
LLM이 만든 pseudo_labeled.jsonl.gz의 라벨 노이즈를 룰 기반으로 정제.

처리 흐름:
  pseudo_labeled.jsonl.gz (raw, ~5000건)
      ↓ 카테고리별 키워드 검증
      ↓ 다중 라벨 정합성 체크
      ↓ 짧은 텍스트 룰
  pseudo_labeled_filtered.jsonl.gz (정제됨)

룰 (관찰된 노이즈 패턴 기반):
  R1: 카테고리별 키워드 검증 — 키워드 0개면 해당 라벨 제거
  R2: 라벨 3개 이상이면 신뢰도 낮은 라벨부터 제거 → 최대 2개
  R3: 짧은 텍스트(<10자)에 라벨 2개 이상 → 가장 신뢰도 높은 1개만
  R4: 모든 라벨 제거되면 is_toxic = False

사용:
  python filter_pseudo_labels.py
  python filter_pseudo_labels.py --diff   # 변경 사항 sample 표시
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IN_PATH  = Path("data/llm_dataset/pseudo_labeled.jsonl.gz")
OUT_PATH = Path("data/llm_dataset/pseudo_labeled_filtered.jsonl.gz")

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech": "혐오발언", "sexual_harassment": "성희롱",
    "sexism": "성차별", "threat": "살해협박", "political": "정치", "other": "기타유해",
}


# ──────────────────────────────────────────────
# 카테고리별 키워드 사전
# ──────────────────────────────────────────────
# 영/한 핵심 키워드. 정규식이라 substring 매칭.
# 키워드 0개면 해당 라벨 제거.

KEYWORDS = {
    "profanity": {
        # 영어 욕설
        "en": [r"\bfuck", r"\bshit", r"\bbitch", r"\bdamn", r"\bcunt", r"\bdick",
               r"\bass\b", r"\bbastard", r"\bidiot", r"\bstupid", r"\bdumb",
               r"\bhell\b", r"\bcrap", r"\bwhore", r"\bslut", r"\bpiss",
               r"\bmoron", r"\bretard", r"\btrash"],
        # 한국어 욕설 (변형 포함)
        "ko": ["씨발", "ㅅㅂ", "씨1발", "ㅆ1발", "ㅆㅂ", "시발", "병신", "ㅂㅅ",
               "개새끼", "새끼", "년", "놈", "지랄", "ㅈㄹ", "좆", "조옺", "씹",
               "꺼져", "닥쳐", "엿먹어", "쓰레기", "ㅈ같", "쟛같", "쟤같",
               "ㅄ", "ㅁㅊ", "미친", "또라이", "또라이"],
    },
    "hate_speech": {
        # 영어 — 인종/종교/LGBTQ/장애 관련
        "en": [r"\bnigger", r"\bnigga", r"\bblack people", r"\bwhite people",
               r"\bjew", r"\bmuslim", r"\bchink", r"\bgook", r"\bspic",
               r"\bfag", r"\bgay", r"\blesbian", r"\bqueer", r"\btranny",
               r"\bretard", r"\bautis", r"\bislam", r"\bchrist",
               r"\bsupremacist", r"\bracist", r"\bracial"],
        # 한국어
        "ko": ["흑인", "백인", "짱깨", "쪽바리", "조센", "조선족", "외노자",
               "동성애", "게이", "레즈", "트랜스", "신천지", "이단",
               "기독교", "개신교", "이슬람", "무슬림", "유대",
               "장애인", "병신", "찐따", "핑이", "헬조선", "탈조선",
               "한남", "한녀", "맘충", "김치녀"],
    },
    "sexual_harassment": {
        "en": [r"\bsex", r"\bporn", r"\bnude", r"\bnaked", r"\brape",
               r"\bpenis", r"\bvagina", r"\bbreast", r"\btit\b", r"\bass\b",
               r"\bfuck me", r"\bhorny", r"\bslut", r"\bwhore", r"\bcum\b"],
        "ko": ["섹스", "성관계", "강간", "성폭행", "야동", "포르노",
               "보지", "자지", "성기", "가슴", "젖", "엉덩이",
               "꼴려", "발기", "사정", "딸딸", "야해",
               "변태", "성희롱", "성추행"],
    },
    "sexism": {
        # 성차별 — 성별 단어가 반드시 있어야 함
        "en": [r"\bwoman", r"\bwomen", r"\bman\b", r"\bmen\b",
               r"\bgirl", r"\bboy", r"\bfemale", r"\bmale\b",
               r"\bfeminis", r"\bsexist", r"\bmisogyn",
               r"\bbitch", r"\bslut", r"\bwhore", r"\bcunt",
               r"\bhousewife", r"\bpregnant", r"\bmother", r"\bfather",
               r"\bgender"],
        "ko": ["여자", "여성", "남자", "남성", "여친", "남친", "와이프", "남편",
               "암컷", "수컷", "년", "놈", "김치녀", "한남", "한녀", "맘충",
               "페미", "성차별", "여혐", "남혐", "보지", "자지",
               "주부", "아내", "엄마", "어머니", "처녀", "총각"],
    },
    "threat": {
        "en": [r"\bkill", r"\bshoot", r"\bgun\b", r"\bmurder", r"\bdie\b",
               r"\bdead", r"\bpoison", r"\bstab", r"\bhurt", r"\bharm",
               r"\battack", r"\bbomb", r"\bburn\b", r"\btorture",
               r"\bbeat (you|him|her|them)", r"\bdestroy",
               r"\bhang", r"\bexterminate", r"\binvade"],
        "ko": ["죽이", "죽일", "죽어", "죽을", "쏴", "쏠", "총", "칼",
               "찌르", "패", "팰", "때려", "협박", "테러", "폭탄",
               "박살", "뽀개", "조져", "감금", "처형", "절단", "잘라",
               "불태", "태워버", "쓸어버", "쳐죽", "뒈져", "뒤져"],
    },
    "political": {
        "en": [r"\btrump", r"\bbiden", r"\bobama", r"\bclinton", r"\breagan",
               r"\bdemocrat", r"\brepublican", r"\bliberal", r"\bconservative",
               r"\bleftis", r"\brightis", r"\bfascis", r"\bcommunis", r"\bsocialis",
               r"\bgovernment", r"\bpresident", r"\bsenat", r"\bcongress",
               r"\bpolitic", r"\belection", r"\bvote\b"],
        "ko": ["이명박", "박근혜", "문재인", "윤석열", "노무현", "김대중",
               "이재명", "안철수", "홍준표", "심상정", "조국",
               "더불어민주당", "국민의힘", "민주당", "정의당", "공산당",
               "좌빨", "보수충", "수꼴", "꼴통", "쥐닭", "문재앙", "이니",
               "정권", "정부", "대통령", "총리", "국회", "선거", "투표",
               "탄핵", "친일", "친북", "빨갱이", "토착왜구"],
    },
    "other": {
        # other는 폭넓은 카테고리라 키워드 매칭 약함.
        # 검증을 가볍게 — 외모/연령/지역 키워드 + 일반 비하 단어
        "en": [r"\bugly", r"\bfat\b", r"\bstupid", r"\bidiot", r"\bdumb",
               r"\bweird", r"\bcreepy", r"\bloser", r"\btrash", r"\bworthless",
               r"\bold\b", r"\bboomer", r"\byoung", r"\bredneck", r"\bhillbilly",
               r"\bhick\b"],
        "ko": ["못생", "오징어", "주름", "뚱뚱", "돼지", "찐따", "쩌리",
               "틀딱", "노친네", "꼰대", "헬조선", "지방", "촌놈", "촌년",
               "전라디언", "경상디언", "충청도", "쓰레기", "병신", "찐따",
               "ㅄ", "한심"],
    },
}


# 카테고리별 신뢰도 (R2에서 라벨 줄일 때 사용)
# 키워드 매칭이 명확한 카테고리가 신뢰도 높음
CATEGORY_CONFIDENCE = {
    "profanity":         3,  # 명확한 욕설 사전
    "threat":            3,  # 명확한 위협 단어
    "political":         3,  # 정치인/정당명 매칭
    "hate_speech":       2,  # 인종/종교/LGBTQ 단어
    "sexism":            2,  # 성별 단어 필수
    "sexual_harassment": 2,  # 성적 단어 필수
    "other":             1,  # 가장 포괄적, 신뢰도 낮음
}


# ──────────────────────────────────────────────
# 필터링 룰
# ──────────────────────────────────────────────

def has_keyword(text: str, category: str, lang: str) -> bool:
    """텍스트에 해당 카테고리의 키워드가 있는지"""
    text_lower = text.lower()
    keywords = KEYWORDS.get(category, {})

    # 영어 키워드는 정규식
    for pat in keywords.get("en", []):
        if re.search(pat, text_lower):
            return True
    # 한국어 키워드는 substring
    for kw in keywords.get("ko", []):
        if kw in text:
            return True
    return False


def filter_record(record: dict) -> tuple[dict, list[str]]:
    """단일 레코드 필터링. 변경된 라벨 목록과 함께 반환."""
    text = record["text"]
    lang = record.get("lang", "")
    labels_dict = record["labels"]
    removed = []

    active_labels = [c for c in CATEGORIES if labels_dict.get(c) == 1]

    # R1: 카테고리별 키워드 검증
    kept_labels = []
    for cat in active_labels:
        if has_keyword(text, cat, lang):
            kept_labels.append(cat)
        else:
            removed.append(f"R1:{cat}")

    # R3: 짧은 텍스트 (<10자) 에 라벨 2개 이상 → 신뢰도 높은 1개만
    meaningful_len = len(re.sub(r"[^\w가-힣]", "", text))
    if meaningful_len < 10 and len(kept_labels) > 1:
        kept_labels.sort(key=lambda c: -CATEGORY_CONFIDENCE.get(c, 0))
        removed.extend([f"R3:{c}" for c in kept_labels[1:]])
        kept_labels = kept_labels[:1]

    # R2: 라벨 3개 이상 → 신뢰도 상위 2개만
    if len(kept_labels) > 2:
        kept_labels.sort(key=lambda c: -CATEGORY_CONFIDENCE.get(c, 0))
        removed.extend([f"R2:{c}" for c in kept_labels[2:]])
        kept_labels = kept_labels[:2]

    # 새 레코드 생성
    new_record = dict(record)
    new_record["labels"] = {c: (1 if c in kept_labels else 0) for c in CATEGORIES}
    new_record["is_toxic"] = len(kept_labels) > 0

    # 라벨이 0개로 줄었으면 toxic_span도 비움
    if not kept_labels:
        new_record["toxic_span"] = ""
        new_record["reason"] = (record.get("reason", "") + " | filtered_clean").strip(" |")[:80]

    return new_record, removed


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", action="store_true", help="변경 사례 20건 미리보기")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("Pseudo Label 필터링")
    logger.info("=" * 60)

    if not IN_PATH.exists():
        logger.error(f"  입력 파일 없음: {IN_PATH}")
        return

    # 로드
    records = []
    with gzip.open(IN_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    logger.info(f"  입력: {len(records):,}건")

    # 필터링
    new_records = []
    removed_counter = Counter()
    changed_examples = []  # (원본, 변경, removed_rules)
    n_changed = 0
    n_full_clean = 0

    for r in records:
        new_r, removed = filter_record(r)
        new_records.append(new_r)
        for rule in removed:
            removed_counter[rule] += 1

        if removed:
            n_changed += 1
            if len(changed_examples) < 20 and any(r["labels"].get(c) for c in CATEGORIES):
                changed_examples.append((r, new_r, removed))
            if not new_r["is_toxic"]:
                n_full_clean += 1

    # 저장
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_size = OUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"\n  필터링 통계:")
    logger.info(f"    변경된 레코드: {n_changed:,}건 ({n_changed/len(records)*100:.1f}%)")
    logger.info(f"    완전 정상화 (toxic→clean): {n_full_clean:,}건")
    logger.info(f"    출력: {OUT_PATH} ({out_size:.2f} MB)")

    logger.info(f"\n  제거된 라벨 (룰별):")
    # R1 (키워드 없음) — 카테고리별
    r1_counts = Counter()
    r2_counts = Counter()
    r3_counts = Counter()
    for rule, cnt in removed_counter.items():
        if rule.startswith("R1:"):
            r1_counts[rule[3:]] += cnt
        elif rule.startswith("R2:"):
            r2_counts[rule[3:]] += cnt
        elif rule.startswith("R3:"):
            r3_counts[rule[3:]] += cnt

    logger.info(f"    R1 (키워드 매칭 실패):")
    for cat in CATEGORIES:
        cnt = r1_counts.get(cat, 0)
        if cnt:
            logger.info(f"      {CATEGORY_KO[cat]:8s}: {cnt:,}건 제거")
    if r2_counts:
        logger.info(f"    R2 (라벨 3개+ 제한):")
        for cat in CATEGORIES:
            cnt = r2_counts.get(cat, 0)
            if cnt:
                logger.info(f"      {CATEGORY_KO[cat]:8s}: {cnt:,}건")
    if r3_counts:
        logger.info(f"    R3 (짧은 텍스트):")
        for cat in CATEGORIES:
            cnt = r3_counts.get(cat, 0)
            if cnt:
                logger.info(f"      {CATEGORY_KO[cat]:8s}: {cnt:,}건")

    # 필터링 전/후 카테고리별 분포
    cat_before = {c: 0 for c in CATEGORIES}
    cat_after  = {c: 0 for c in CATEGORIES}
    n_toxic_before = 0
    n_toxic_after = 0
    for old, new in zip(records, new_records):
        if old["is_toxic"]: n_toxic_before += 1
        if new["is_toxic"]: n_toxic_after  += 1
        for c in CATEGORIES:
            cat_before[c] += old["labels"].get(c, 0)
            cat_after[c]  += new["labels"].get(c, 0)

    logger.info(f"\n  카테고리별 분포 변화:")
    logger.info(f"    {'카테고리':12s} {'before':>8s} {'after':>8s} {'-/+':>8s}")
    for c in CATEGORIES:
        diff = cat_after[c] - cat_before[c]
        logger.info(f"    {CATEGORY_KO[c]:12s} {cat_before[c]:>8d} {cat_after[c]:>8d} {diff:>+8d}")
    logger.info(f"    {'─'*45}")
    logger.info(f"    {'유해 합계':12s} {n_toxic_before:>8d} {n_toxic_after:>8d} {n_toxic_after-n_toxic_before:>+8d}")
    logger.info(f"    ({n_toxic_before/len(records)*100:.1f}% → {n_toxic_after/len(records)*100:.1f}%)")

    # 변경 사례 미리보기
    if args.diff and changed_examples:
        logger.info(f"\n  변경 사례 (최대 20건):")
        for i, (old, new, removed) in enumerate(changed_examples):
            old_labels = [CATEGORY_KO[c] for c in CATEGORIES if old["labels"].get(c)]
            new_labels = [CATEGORY_KO[c] for c in CATEGORIES if new["labels"].get(c)]
            logger.info(f"\n  [{i+1}] [{old.get('lang','?')}] {old['text'][:80]}")
            logger.info(f"      before: {old_labels}")
            logger.info(f"      after : {new_labels}")
            logger.info(f"      removed by: {removed}")


if __name__ == "__main__":
    main()
