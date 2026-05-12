"""
llm_pseudo_label_v4.py
======================
3B + 1-stage 구조 (HateCoT 통합 + SlangLLM 사전 차단)

설계 변경:
  v3 (3-stage HateCoT)에서 누적 오류 + 환각 문제 발견 → 1-stage로 회귀
  
  - HateCoT: 5단계 reasoning을 한 프롬프트 안에 chain-of-thought로 풀어쓰기
    (별도 호출 X, in-prompt CoT)
  - SlangLLM: PoS 점수로 사전 차단 → 정상 후보는 LLM 호출 안 함
    (속도↑, false positive↓)

논문:
  [1] HateGuard (Ko et al., 2312.15099) — HateCoT 5단계
      Target → Derogation → Direction → Incitation → Decision
  [2] Patel & Alsobeh (SlangLLM, 2025) — PoS 점수 기반 사전 필터링
      INTJ:1.0 ADJ:0.8 VERB:0.7 PROPN:0.6 NOUN:0.5 → max < threshold면 정상으로 패스

카테고리 (7개):
  profanity / hate_speech / sexual_harassment / sexism / threat / political / other
"""

from __future__ import annotations

import os
import re
import json
import gzip
import time
import logging
import sys
import requests
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from slang_pos_scorer import score_tokens, format_for_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3.2:3b"
IN_PATH     = Path("data/llm_dataset/pseudo_target.json")
OUT_PATH    = Path("data/llm_dataset/pseudo_labeled.jsonl.gz")
BATCH_SIZE  = 50
MAX_SAMPLES = 5000   # 시범. 검증 후 5000으로
USE_GZIP    = True
MAX_REASON_LEN = 50
MAX_SPAN_LEN   = 30

# SlangLLM 사전 차단 임계값:
#   토큰 중 최대 PoS 점수 < SKIP_THRESHOLD → LLM 호출 없이 정상 처리
#   PoS 점수: INTJ=1.0, ADJ=0.8, VERB=0.7, PROPN=0.6, NOUN=0.5, 기타=0.2
#   0.6 이상이면 일단 한번 LLM에 보내 검토 (보수적)
SKIP_THRESHOLD = 0.6
USE_POS_SCORING = True

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity":         "욕설",
    "hate_speech":       "혐오발언",
    "sexual_harassment": "성희롱",
    "sexism":            "성차별",
    "threat":            "살해협박",
    "political":         "정치",
    "other":             "기타유해",
}


# ──────────────────────────────────────────────
# 프롬프트 (1-stage, HateCoT in-prompt)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a content moderation classifier.

Categories (multi-label, [] if benign):
- profanity: vulgar/swear words
- hate_speech: attacks based on race/ethnicity/religion/nationality/LGBTQ/disability
- sexual_harassment: explicit sexual insults or content
- sexism: explicit gender-based attacks
- threat: explicit threats of violence/harm/death
- political: attacks on politicians, parties, or political ideology groups
- other: appearance shaming, ageism, regional, or other clear toxicity not above

⚠ Rules:
- Label what is EXPLICITLY in the text. Do not infer.
- Common conversation (questions, greetings, opinions) is NOT toxic.
- Empty list [] is a valid and common answer for normal text.

Think briefly through these steps (do not output them, just use them):
  1) Is there a target (person/group/identity)?
  2) Are there derogatory or harmful words explicitly?
  3) Are those words directed at the target?
  4) Do they propose or incite hate/harm?
  5) Decision: pick categories or [].

Then output JSON only:
{"labels":["..."],"toxic_span":"<=30 chars","reason":"<=50 chars"}

The toxic_span MUST be a literal substring of the text. If you cannot find toxic words in the text, return "".
"""

USER_TEMPLATE = """Text: "{text}"
Paragraph context: "{paragraph}"
Notable tokens: {pos_hint}

Output JSON:"""


# 거부 방어
REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i won't", "i will not", "i'm not able", "i am not able",
    "as an ai", "as a language model",
    "i'm sorry", "i apologize", "against my",
    "inappropriate", "not appropriate",
    "죄송", "할 수 없", "도와드릴 수 없",
]

REINFORCEMENT_PREFIX = """⚠ Academic research context.
You are CLASSIFYING pre-collected dataset samples. NOT generating harmful content.
Refusing to classify is a failure. Output JSON only. If unsure, output {"labels":[],"toxic_span":"","reason":"uncertain"}.

"""


# ──────────────────────────────────────────────
# 유틸 — 사전 차단 + LLM 호출
# ──────────────────────────────────────────────

def _is_meaningless(text: str) -> bool:
    """텍스트가 사실상 의미 없는지 (구두점만, 너무 짧음)"""
    if not text:
        return True
    meaningful = re.sub(r"[^\w가-힣]", "", text)
    return len(meaningful) < 3


def _max_pos_score(scored: list[tuple[str, str, float]]) -> float:
    """PoS 점수 중 최대값. 텍스트가 비었으면 0."""
    if not scored:
        return 0.0
    return max(s for _, _, s in scored)


def _ollama_request(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
    full = f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":  MODEL,
                    "prompt": full,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 150,
                        "top_p": 0.9,
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"  Ollama 호출 실패: {e}")
    return ""


def _try_parse(raw: str) -> dict | None:
    if not raw or "{" not in raw or "}" not in raw:
        return None
    try:
        s, e = raw.index("{"), raw.rindex("}") + 1
        parsed = json.loads(raw[s:e])
        labels = [l for l in parsed.get("labels", []) if l in CATEGORIES]
        return {
            "labels":     labels,
            "toxic_span": str(parsed.get("toxic_span", ""))[:MAX_SPAN_LEN],
            "reason":     str(parsed.get("reason", ""))[:MAX_REASON_LEN],
        }
    except (json.JSONDecodeError, ValueError):
        return None


def _looks_refused(raw: str) -> bool:
    if not raw:
        return True
    low = raw.lower()
    return any(p in low for p in REFUSAL_PATTERNS)


def call_llm(text: str, paragraph: str, pos_hint: str) -> dict:
    """1회 호출. 거부 감지 시 2차 강화 프롬프트 재시도."""
    user_prompt = USER_TEMPLATE.format(
        text=text, paragraph=paragraph, pos_hint=pos_hint or "(none)",
    )

    # 1차
    raw1 = _ollama_request(SYSTEM_PROMPT, user_prompt)
    parsed = _try_parse(raw1)
    if parsed is not None and not _looks_refused(raw1):
        return parsed

    # 2차 (거부 감지 시)
    raw2 = _ollama_request(REINFORCEMENT_PREFIX + SYSTEM_PROMPT, user_prompt)
    parsed2 = _try_parse(raw2)
    if parsed2 is not None:
        if not parsed2["reason"]:
            parsed2["reason"] = "refused_retry_ok"
        return parsed2

    # 모두 실패
    if _looks_refused(raw2):
        return {"labels": [], "toxic_span": "", "reason": "refused_final"}
    return {"labels": [], "toxic_span": "", "reason": "parse_error"}


# ──────────────────────────────────────────────
# 메인 추론 함수
# ──────────────────────────────────────────────

def classify(record: dict) -> dict:
    """1-stage 분류 + SlangLLM 사전 차단 + 환각 검증."""
    text = record["text"]
    paragraph = record.get("full_text") or record.get("context", "") + " " + text
    paragraph = paragraph.strip()[:600]
    lang = record.get("lang", "en")

    # 사전 필터 1: 의미 없는 짧은 텍스트
    if _is_meaningless(text):
        return {"labels": [], "toxic_span": "", "reason": "too_short"}

    # 사전 필터 2: SlangLLM PoS 점수
    # 슬랭/욕설/감정어가 전혀 없으면 LLM 호출 없이 정상 처리 (속도↑)
    scored = []
    pos_hint = ""
    if USE_POS_SCORING:
        scored = score_tokens(text, lang=lang, top_k=8)
        max_score = _max_pos_score(scored)
        if max_score < SKIP_THRESHOLD:
            # 모든 토큰의 PoS 점수가 낮음 → 슬랭/욕설/공격성 표현 없음 → 정상
            return {"labels": [], "toxic_span": "", "reason": "pos_skip"}
        pos_hint = format_for_prompt(scored)

    # LLM 호출
    out = call_llm(text, paragraph, pos_hint)

    # 환각 검증: toxic_span이 실제 텍스트에 있는지 확인
    span = out["toxic_span"]
    if out["labels"] and span:
        text_norm = "".join(text.split()).lower()
        span_norm = "".join(span.split()).lower()
        if span_norm and span_norm not in text_norm:
            # span이 텍스트에 없음 → 환각 → 라벨 무효
            return {"labels": [], "toxic_span": "", "reason": "hallucinated_span"}

    return out


# ──────────────────────────────────────────────
# JSONL I/O
# ──────────────────────────────────────────────

def open_jsonl_write(path: Path, mode: str = "at"):
    if USE_GZIP and str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def open_jsonl_read(path: Path):
    if USE_GZIP and str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open_jsonl_read(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def make_compact_record(record: dict, llm_out: dict) -> dict:
    return {
        "id":         record["id"],
        "text":       record["text"],
        "context":    record.get("context", "")[:200],
        "lang":       record.get("lang", ""),
        "labels":     {c: (1 if c in llm_out["labels"] else 0) for c in CATEGORIES},
        "is_toxic":   len(llm_out["labels"]) > 0,
        "toxic_span": llm_out["toxic_span"],
        "reason":     llm_out["reason"],
    }


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("[3단계] LLM Pseudo Labeling v4 (3B + 1-stage + SlangLLM 차단)")
    logger.info(f"  ⚠ MAX_SAMPLES = {MAX_SAMPLES}")
    logger.info(f"  PoS 사전 차단 임계값: {SKIP_THRESHOLD}")
    logger.info("=" * 60)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(IN_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"  입력 레코드: {len(records):,}건")

    done_ids = load_done_ids(OUT_PATH)
    if done_ids:
        logger.info(f"  이미 처리됨 (재시작): {len(done_ids):,}건")

    todo = [r for r in records if r["id"] not in done_ids]
    if MAX_SAMPLES:
        remain = max(0, MAX_SAMPLES - len(done_ids))
        todo = todo[:remain]

    logger.info(f"  처리할 레코드: {len(todo):,}건")
    logger.info(f"  모델: {MODEL}")
    logger.info(f"  출력: {OUT_PATH}")
    logger.info(f"  카테고리: {len(CATEGORIES)}개")

    if not todo:
        logger.info("  처리할 레코드가 없습니다.")
        return

    # 사전 점검
    try:
        tags_resp = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=5)
        tags_resp.raise_for_status()
        models = [m.get("name", "") for m in tags_resp.json().get("models", [])]
        logger.info(f"  Ollama 서버 OK. 모델: {models[:5]}")
        if MODEL not in models and not any(MODEL in m for m in models):
            logger.error(f"  ✗ 모델 '{MODEL}' 설치 안 됨. `ollama pull {MODEL}` 실행하세요.")
            return

        # 실제 추론 테스트
        test_rec = {"id": "test", "text": "I hate you", "lang": "en", "full_text": "I hate you"}
        test_out = classify(test_rec)
        if test_out.get("reason", "").endswith("_fail"):
            logger.error(f"  ✗ 테스트 추론 실패: {test_out}")
            return
        logger.info(f"  연결 OK: {test_out}")
    except requests.exceptions.ConnectionError:
        logger.error(f"  ✗ Ollama 서버 연결 실패: {OLLAMA_URL}")
        logger.error(f"     `ollama serve` 로 서버를 먼저 띄우세요.")
        return
    except Exception as e:
        logger.error(f"  ✗ 연결 실패: {e}")
        return

    t_start = time.time()
    n_toxic = 0
    reason_counts = {"too_short": 0, "pos_skip": 0, "hallucinated_span": 0,
                     "refused_final": 0, "parse_error": 0}
    buffer = []

    with open_jsonl_write(OUT_PATH, "at") as fout:
        for i, record in enumerate(tqdm(todo, desc="Pseudo Labeling")):
            llm_out = classify(record)
            reason = llm_out.get("reason", "")
            if reason in reason_counts:
                reason_counts[reason] += 1

            compact = make_compact_record(record, llm_out)
            if compact["is_toxic"]:
                n_toxic += 1

            buffer.append(json.dumps(compact, ensure_ascii=False))

            if len(buffer) >= BATCH_SIZE:
                fout.write("\n".join(buffer) + "\n")
                fout.flush()
                buffer.clear()

                elapsed = time.time() - t_start
                speed = (i + 1) / elapsed
                remain = (len(todo) - i - 1) / max(speed, 0.01) / 60
                logger.info(f"  [{i+1}/{len(todo)}] flush | {speed:.2f}건/초 | ~{remain:.0f}분 남음")

        if buffer:
            fout.write("\n".join(buffer) + "\n")

    total_elapsed = time.time() - t_start
    file_size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"\n완료!")
    logger.info(f"  소요 시간: {total_elapsed/60:.1f}분")
    logger.info(f"  처리 속도: {len(todo)/max(total_elapsed,0.01):.2f}건/초")
    logger.info(f"  유해(이번): {n_toxic:,}건")
    logger.info(f"  파일 크기: {file_size_mb:.2f} MB")
    logger.info(f"  특수 처리:")
    for k, v in reason_counts.items():
        logger.info(f"    {k:20s}: {v}건")

    if MAX_SAMPLES and len(todo) > 0:
        speed = len(todo) / max(total_elapsed, 0.01)
        full_time_h = 5000 / speed / 3600
        logger.info(f"\n  → 5000건 풀 실행 예상 시간: {full_time_h:.1f}시간")

    # 카테고리별 통계
    cat_counts = {c: 0 for c in CATEGORIES}
    n_total, n_toxic_total = 0, 0
    with open_jsonl_read(OUT_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                n_total += 1
                if r["is_toxic"]:
                    n_toxic_total += 1
                for c in CATEGORIES:
                    if r["labels"].get(c) == 1:
                        cat_counts[c] += 1
            except Exception:
                continue

    logger.info(f"\n전체 통계:")
    logger.info(f"  유해/정상: {n_toxic_total:,} / {n_total - n_toxic_total:,}")
    for c in CATEGORIES:
        logger.info(f"    {CATEGORY_KO[c]:10s}: {cat_counts[c]:,}건")


if __name__ == "__main__":
    main()
