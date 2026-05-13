"""
postprocess_cleanup_v2.py
==========================
postprocess_cleanup.py 의 R5를 강화한 버전.

변경점:
  - R5 강화: sexism 제거 기준을 (text 전체 성별 키워드 존재)가 아니라
            (Q1 detail에 성별 언급 + Q2 terms에 성별 비하어) AND 로직으로
  - R7 신규: sexism 있는데 Q2 terms에 성별 비하어 없으면 제거

여성/남성 비하어 사전 추가.

실행 전제:
  - 이미 postprocess_cleanup.py 한 번 돌린 결과 위에 추가 정제
  - 혹은 v3_preCleanup 백업에서 재시작도 가능 (REPROCESS_FROM_BACKUP=True)
"""

import json
import logging
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from poison_level import compute_poison_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IN_PATH     = Path("data/llm_dataset/pseudo_labeled.json")
OUT_PATH    = Path("data/llm_dataset/pseudo_labeled.json")
BACKUP_PATH = Path("data/llm_dataset/pseudo_labeled_v4_preCleanupV2.json")

# True로 바꾸면 v3_preCleanup 백업 기준으로 재시작 (R1~R6 다시 적용)
REPROCESS_FROM_BACKUP = False
V3_BACKUP = Path("data/llm_dataset/pseudo_labeled_v3_preCleanup.json")

CATEGORIES = ["profanity", "hate_speech", "sexual_harassment", "sexism", "threat"]
CATEGORY_KO = {
    "profanity":         "욕설",
    "hate_speech":       "인종차별",
    "sexual_harassment": "성희롱",
    "sexism":            "성차별",
    "threat":            "살해협박",
}

# ──────────────────────────────────────────────
# 성별 관련 키워드/비하어 사전
# ──────────────────────────────────────────────

# 성별 지시어 (he, she, woman, ...)
GENDER_REFERENTS_EN = {
    "woman", "women", "girl", "girls", "female", "feminine", "she", "her", "hers", "herself",
    "lady", "ladies", "wife", "wives", "mother", "mom", "mommy", "mama", "sister", "daughter",
    "aunt", "grandmother", "granny", "madam", "miss", "mrs", "ms",
    "man", "men", "boy", "boys", "male", "masculine", "he", "him", "his", "himself",
    "guy", "guys", "dude", "dudes", "husband", "father", "dad", "daddy", "papa", "brother", "son",
    "uncle", "grandfather", "grandpa", "sir", "mister", "mr",
}
GENDER_REFERENTS_KO = {
    "여자", "여성", "여성분", "여자분", "녀", "아내", "부인", "어머니", "엄마", "누나", "언니",
    "딸", "이모", "고모", "할머니", "아주머니", "아줌마",
    "남자", "남성", "남성분", "남자분", "남", "남편", "아버지", "아빠", "형", "오빠",
    "아들", "삼촌", "할아버지", "아저씨",
}
GENDER_REFERENTS = {w.lower() for w in GENDER_REFERENTS_EN} | GENDER_REFERENTS_KO

# 여성 대상 비하어 (context 따라 오용)
FEMALE_SLURS_EN = {
    "bitch", "bitches", "bitchy",
    "slut", "sluts", "slutty",
    "whore", "whores",
    "cunt", "cunts",
    "pussy", "pussies",
    "cow", "cows",   # "lazy cow" 같이 여성 비하
    "hag", "hags",
    "nag", "nags",
    "tramp", "tramps",
    "skank", "skanks",
    "thot", "thots",
    "bimbo", "bimbos",
    "broad", "broads",
    "karen", "karens",
}
FEMALE_SLURS_KO = {
    "년", "걸레", "보지",
    "맘충", "김치녀", "된장녀",
}

MALE_SLURS_EN = {
    "dick", "dicks",
    "prick", "pricks",
    "douche", "douchebag",
    "bastard", "bastards",
    "simp", "simps",
    "incel", "incels",
    "fuckboy",
}
MALE_SLURS_KO = {
    "놈", "새끼", "자지", "한남",
}

GENDER_SLURS = (
    {w.lower() for w in FEMALE_SLURS_EN} | FEMALE_SLURS_KO |
    {w.lower() for w in MALE_SLURS_EN} | MALE_SLURS_KO
)

# 성역할/고정관념 강요 문구 (phrase level)
GENDER_ROLE_PATTERNS = [
    r"\bwomen\s+(should|shouldn't|belong|must|can't|cannot|are\s+supposed)",
    r"\bmen\s+(should|shouldn't|don't\s+cry|are\s+supposed)",
    r"\bgirls?\s+(should|shouldn't|belong)",
    r"\bboys?\s+(will\s+be\s+boys|don't\s+cry)",
    r"\ba\s+woman's\s+place",
    r"\bman\s+of\s+the\s+house",
    r"\bstay\s+at\s+home\s+(mom|wife)",
    r"여자는\s+(집에|살림|애)",
    r"남자는\s+(울면|남자답게)",
]
ROLE_RE = re.compile("|".join(GENDER_ROLE_PATTERNS), re.IGNORECASE)


# ──────────────────────────────────────────────
# 기존 규칙 (R1-R3) 재탑재
# ──────────────────────────────────────────────

NEGATION_PATTERNS = [
    r"\bno\s+derogatory",
    r"\bdoes\s+not\s+incite",
    r"\bdoes\s+not\s+contain",
    r"\bnot\s+hateful",
    r"\bnot\s+toxic",
    r"\bno\s+\w+\s+terms",
    r"\bdo\s+not\s+incite",
    r"\bno\s+incitement",
    r"\bis\s+not\s+directed",
    r"\blikely\s+being\s+used\s+to\s+express",
    r"\bno\s+target",
    r"\bin\s+this\s+context[^.]*(likely|benign|neutral|harmless)",
]
NEG_RE = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

BENIGN_TOKENS = {
    "right", "too", "again", "everyone", "something", "anything",
    "really", "just", "some", "much", "very", "so", "all",
    "maybe", "probably", "actually", "seriously", "honestly",
    "good", "bad", "nice", "fine", "okay", "ok", "well",
    "people", "person", "someone", "anyone", "them", "they",
    "want", "need", "think", "know", "like", "love",
    "status", "regardless",
}


def get_active_categories(record) -> list:
    labels = record.get("labels", {}) or {}
    return [c for c in CATEGORIES if labels.get(c) == 1]


def has_gender_referent(text: str) -> bool:
    """text에 성별 지시어가 있는가?"""
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    return any(t in GENDER_REFERENTS for t in tokens)


def has_gender_slur(text: str) -> bool:
    """text에 성별 비하어가 있는가?"""
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    return any(t in GENDER_SLURS for t in tokens)


def has_gender_role_phrase(text: str) -> bool:
    """성역할 강요 문구가 있는가?"""
    return bool(ROLE_RE.search(text))


# ──────────────────────────────────────────────
# 규칙 적용
# ──────────────────────────────────────────────

def should_strip_all(record) -> tuple:
    """R1-R3: 전체 제거 규칙"""
    cot = record.get("cot_steps", {})
    q1 = cot.get("q1_target", {}).get("answer", "no").lower()
    q2 = cot.get("q2_derogation", {}).get("answer", "no").lower()
    reason = str(record.get("reason", "") or cot.get("q5_decision", {}).get("reason", "") or "")
    toxic_span = str(record.get("toxic_span", "") or "").strip().strip('"\'.,!?')
    active = get_active_categories(record)

    if not active:
        return False, ""

    if q1 == "no" and q2 == "no":
        return True, "R1_no_target_no_derog"

    if reason and NEG_RE.search(reason):
        return True, "R2_reason_contradicts"

    if toxic_span:
        toks = toxic_span.lower().split()
        if len(toks) <= 2 and all(t.strip(".,!?;:") in BENIGN_TOKENS for t in toks):
            return True, "R3_benign_span"

    return False, ""


def should_strip_weak_sexism(record) -> tuple:
    """
    R5-v2 + R7: sexism 거짓 양성 제거.

    Keep sexism iff 다음 중 하나:
      A. Q1 detail에 성별 지시어 AND (Q2 terms에 성별 비하어 OR text에 성역할 강요)
      B. text/context에 성별 비하어 AND 성별 지시어
      C. text에 성역할 강요 문구
      D. Q1 detail에 성별 지시어 있음 (gender 명시적 타겟)
      E. Q2 terms에 성별 비하어 있음 (explicit slur)

    위 조건 아닌 sexism은 ghost로 간주.
    """
    active = get_active_categories(record)
    if "sexism" not in active:
        return False, ""

    text = (record.get("text", "") + " " + record.get("context", "")).lower()
    cot = record.get("cot_steps", {})
    q1_detail = cot.get("q1_target", {}).get("detail", "").lower()
    q2_terms = " ".join(str(t) for t in cot.get("q2_derogation", {}).get("terms", [])).lower()

    detail_has_gender = has_gender_referent(q1_detail)
    terms_has_slur    = has_gender_slur(q2_terms)
    text_has_role     = has_gender_role_phrase(text)
    text_has_slur     = has_gender_slur(text)
    text_has_referent = has_gender_referent(text)

    # A
    if detail_has_gender and (terms_has_slur or text_has_role):
        return False, "A_detail_gender+signal"
    # B
    if text_has_slur and text_has_referent:
        return False, "B_text_slur+referent"
    # C
    if text_has_role:
        return False, "C_role_phrase"
    # D: Q1 detail이 명시적으로 성별을 가리킴 (예: 'her', 'she', 'women')
    if detail_has_gender:
        return False, "D_detail_gender_only"
    # E: terms에 explicit slur
    if terms_has_slur:
        return False, "E_terms_slur_only"

    return True, "R5v2_weak_sexism"


def should_strip_weak_hate_speech(record) -> bool:
    """기존 R6 동일"""
    active = get_active_categories(record)
    if "hate_speech" not in active:
        return False

    text = (record.get("text", "") + " " + record.get("context", "")).lower()
    identity_keywords = [
        "black", "white", "asian", "jew", "jewish", "arab", "hispanic", "latino", "african",
        "chinese", "japanese", "korean", "mexican", "indian", "muslim", "christian", "hindu",
        "race", "racist", "ethnic", "nationality", "immigrant", "refugee", "foreigner",
        "gay", "lesbian", "lgbt", "queer", "trans", "homosexual",
        "disabled", "retard", "crippl",
        "흑인", "백인", "조선족", "중국인", "일본인", "유대인", "무슬림",
        "외국인", "이민자", "난민", "동성애", "장애인",
    ]
    if any(kw in text for kw in identity_keywords):
        return False

    cot = record.get("cot_steps", {})
    if cot.get("q1_target", {}).get("target_type", "") == "identity":
        detail = cot.get("q1_target", {}).get("detail", "").lower()
        if detail and detail != "none":
            return False

    return True


def apply_cleanup_v2(record: dict) -> dict:
    """v2 규칙 적용"""
    original_cats = get_active_categories(record)
    if not original_cats:
        record["_cleanup_v2_applied"] = []
        return record

    applied_rules = []
    new_cats = list(original_cats)

    strip_all, rule = should_strip_all(record)
    if strip_all:
        new_cats = []
        applied_rules.append(rule)
    else:
        # R5-v2: 강화된 sexism 제거
        strip_sex, reason = should_strip_weak_sexism(record)
        if strip_sex:
            new_cats = [c for c in new_cats if c != "sexism"]
            applied_rules.append(reason)

        # R6: hate_speech 제거
        if should_strip_weak_hate_speech(record):
            new_cats = [c for c in new_cats if c != "hate_speech"]
            applied_rules.append("R6_weak_hate_speech")

    # labels / label_names 재작성
    record["labels"] = {c: (1 if c in new_cats else 0) for c in CATEGORIES}
    record["label_names"] = [CATEGORY_KO[c] for c in new_cats]
    record["is_toxic"] = len(new_cats) > 0

    if not new_cats:
        record["toxic_span"] = ""

    # poison_level / action 재계산
    slang_conf = record.get("slang_analysis", {}).get("slang_confidence", 0.0)
    cot_conf = record.get("cot_steps", {}).get("q5_decision", {}).get("confidence", 0.0)
    if not new_cats and original_cats:
        cot_conf = min(cot_conf, 0.2)
    poison = compute_poison_level(
        slang_confidence=slang_conf,
        cot_confidence=cot_conf,
        categories=new_cats,
    )
    record.update(poison.to_dict())

    if poison.action in ("BLOCK", "FILTER"):
        record["is_toxic"] = True

    record["_cleanup_v2_applied"] = applied_rules
    return record


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("[후처리 v2] R5 강화 — sexism ghost 공격적 제거")
    logger.info("=" * 60)

    # 입력 소스 결정
    if REPROCESS_FROM_BACKUP and V3_BACKUP.exists():
        logger.info(f"  재시작 모드: v3_preCleanup 백업에서 시작")
        src = V3_BACKUP
    else:
        src = IN_PATH
    if not src.exists():
        logger.error(f"입력 파일 없음: {src}")
        return

    # 백업
    if not BACKUP_PATH.exists():
        logger.info(f"  백업 생성: {BACKUP_PATH}")
        BACKUP_PATH.write_bytes(IN_PATH.read_bytes())

    with open(src, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"  로드: {len(records):,}건 (source={src.name})")

    # Before 통계
    before_cats = Counter()
    before_actions = Counter()
    for r in records:
        for c in get_active_categories(r):
            before_cats[c] += 1
        before_actions[r.get("action", "PASS")] += 1

    # 적용
    logger.info("\n[규칙 적용]")
    rule_counter = Counter()
    for r in tqdm(records, desc="Cleanup v2"):
        apply_cleanup_v2(r)
        for rule in r.get("_cleanup_v2_applied", []):
            rule_counter[rule] += 1

    # 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # After 통계
    after_cats = Counter()
    after_actions = Counter()
    for r in records:
        for c in get_active_categories(r):
            after_cats[c] += 1
        after_actions[r.get("action", "PASS")] += 1

    n = len(records)
    logger.info(f"\n{'='*60}")
    logger.info("완료!")
    logger.info(f"{'='*60}")

    logger.info("\n  규칙 적용 빈도:")
    for rule, cnt in rule_counter.most_common():
        logger.info(f"    {rule:30s}: {cnt:,}건")

    logger.info("\n  카테고리 분포 (Before → After):")
    for c in CATEGORIES:
        b = before_cats[c]
        a = after_cats[c]
        delta = a - b
        sign = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        pct = a / n * 100
        logger.info(f"    {c:20s}: {b:5,} → {a:5,} ({pct:4.1f}%)  ({sign}{abs(delta):,})")

    logger.info("\n  Action 분포 (Before → After):")
    for act in ["BLOCK", "FILTER", "WARN", "PASS"]:
        b = before_actions[act]
        a = after_actions[act]
        delta = a - b
        sign = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        logger.info(f"    {act:7s}: {b:5,} → {a:5,}  ({sign}{abs(delta):,})")


if __name__ == "__main__":
    main()
