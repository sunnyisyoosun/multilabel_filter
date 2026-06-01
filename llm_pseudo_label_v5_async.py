"""
llm_pseudo_label_v5_async.py
============================
v5의 async 고속화 버전.

변경점:
  - AsyncOpenAI 사용
  - CONCURRENCY(기본 64)개씩 동시 요청 (asyncio.gather + Semaphore)
  - SlangLLM 사전차단/환각검증/재시작은 v5와 동일
  - 사전차단(pos_skip, too_short)은 LLM 호출 없이 동기 처리 → API 부하 감소

논문:
  [1] HateGuard (2312.15099) — HateCoT 5단계 in-prompt CoT
  [2] Patel & Alsobeh (SlangLLM, 2025) — PoS 점수 기반 사전 필터링

카테고리 (6개):
  profanity / hate_speech / gender / threat / political / other
"""

from __future__ import annotations

import os
import re
import json
import gzip
import time
import asyncio
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from slang_pos_scorer import score_tokens, format_for_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
API_BASE = "http://203.255.188.62:8002/v1"
API_KEY  = "advanced_ba"
MODEL    = "google/gemma-4-31B-it"
IN_PATH     = Path("data/llm_dataset/pseudo_target.json")
OUT_PATH    = Path("data/llm_dataset/pseudo_labeled_openAI.jsonl.gz")

CONCURRENCY = 64      # 동시 요청 수
FLUSH_EVERY = 200     # N건마다 파일에 기록
MAX_SAMPLES = 2000    # 시범. 검증 후 5000으로
USE_GZIP    = True
MAX_REASON_LEN = 50
MAX_SPAN_LEN   = 30
REQUEST_TIMEOUT = 60  # 초

_client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY, timeout=REQUEST_TIMEOUT)

SKIP_THRESHOLD = 0.6
USE_POS_SCORING = True

CATEGORIES = [
    "profanity", "hate_speech", "gender", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity":         "욕설",
    "hate_speech":       "혐오발언",
    "gender":            "성 관련",
    "threat":            "살해협박",
    "political":         "정치",
    "other":             "기타유해",
}


# ──────────────────────────────────────────────
# 프롬프트 (v5와 동일)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a content moderation classifier.

Categories (multi-label, [] if benign):
- profanity: vulgar/swear words
- hate_speech: attacks based on race/ethnicity/religion/nationality/LGBTQ/disability
- gender: gender-based attacks, sexism, or explicit sexual insults/content
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
# 유틸 (동기 — 사전차단/파싱/검증)
# ──────────────────────────────────────────────

def _is_meaningless(text: str) -> bool:
    if not text:
        return True
    meaningful = re.sub(r"[^\w가-힣]", "", text)
    return len(meaningful) < 3


def _max_pos_score(scored) -> float:
    if not scored:
        return 0.0
    return max(s for _, _, s in scored)


def _try_parse(raw: str):
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


# ──────────────────────────────────────────────
# async LLM 호출
# ──────────────────────────────────────────────

async def _async_request(system_prompt: str, user_prompt: str, retries: int = 2) -> str:
    for attempt in range(retries):
        try:
            resp = await _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"  API 호출 실패: {e}")
            else:
                await asyncio.sleep(1.0)
    return ""


async def call_llm_async(text: str, paragraph: str, pos_hint: str) -> dict:
    """1회 호출 + 거부 시 강화 재시도 (async)."""
    user_prompt = USER_TEMPLATE.format(
        text=text, paragraph=paragraph, pos_hint=pos_hint or "(none)",
    )
    raw1 = await _async_request(SYSTEM_PROMPT, user_prompt)
    parsed = _try_parse(raw1)
    if parsed is not None and not _looks_refused(raw1):
        return parsed

    raw2 = await _async_request(REINFORCEMENT_PREFIX + SYSTEM_PROMPT, user_prompt)
    parsed2 = _try_parse(raw2)
    if parsed2 is not None:
        if not parsed2["reason"]:
            parsed2["reason"] = "refused_retry_ok"
        return parsed2

    if _looks_refused(raw2):
        return {"labels": [], "toxic_span": "", "reason": "refused_final"}
    return {"labels": [], "toxic_span": "", "reason": "parse_error"}


# ──────────────────────────────────────────────
# 분류 (사전차단은 동기, LLM만 async)
# ──────────────────────────────────────────────

def presift(record: dict):
    """LLM 호출 전 사전 판단. (결과dict, pos_hint) 반환.
    결과dict가 None이면 LLM 호출 필요."""
    text = record["text"]
    lang = record.get("lang", "en")

    if _is_meaningless(text):
        return {"labels": [], "toxic_span": "", "reason": "too_short"}, ""

    pos_hint = ""
    if USE_POS_SCORING:
        scored = score_tokens(text, lang=lang, top_k=8)
        if _max_pos_score(scored) < SKIP_THRESHOLD:
            return {"labels": [], "toxic_span": "", "reason": "pos_skip"}, ""
        pos_hint = format_for_prompt(scored)

    return None, pos_hint  # LLM 호출 필요


def verify_hallucination(text: str, out: dict) -> dict:
    """toxic_span이 실제 텍스트에 있는지 검증."""
    span = out.get("toxic_span", "")
    if out.get("labels") and span:
        text_norm = "".join(text.split()).lower()
        span_norm = "".join(span.split()).lower()
        if span_norm and span_norm not in text_norm:
            return {"labels": [], "toxic_span": "", "reason": "hallucinated_span"}
    return out


async def classify_async(record: dict, sem: asyncio.Semaphore) -> dict:
    """사전차단 후 필요시 LLM 호출 (semaphore로 동시성 제한)."""
    text = record["text"]
    paragraph = (record.get("full_text") or record.get("context", "") + " " + text)
    paragraph = paragraph.strip()[:600]

    presift_out, pos_hint = presift(record)
    if presift_out is not None:
        return presift_out  # LLM 호출 안 함

    async with sem:
        out = await call_llm_async(text, paragraph, pos_hint)
    return verify_hallucination(text, out)


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
# 메인 (async)
# ──────────────────────────────────────────────

async def run():
    logger.info("=" * 60)
    logger.info("[3단계] LLM Pseudo Labeling v5-async (gemma-31B + 동시요청)")
    logger.info(f"  동시성: {CONCURRENCY}  |  MAX_SAMPLES: {MAX_SAMPLES}")
    logger.info(f"  PoS 사전차단 임계값: {SKIP_THRESHOLD}")
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
    logger.info(f"  처리할 레코드: {len(todo):,}건  |  모델: {MODEL}")

    if not todo:
        logger.info("  처리할 레코드가 없습니다.")
        return

    # 사전 점검 (1건 테스트)
    try:
        sem_test = asyncio.Semaphore(1)
        test_rec = {"id": "test", "text": "I hate you", "lang": "en", "full_text": "I hate you"}
        test_out = await classify_async(test_rec, sem_test)
        logger.info(f"  연결 OK: {test_out}")
    except Exception as e:
        logger.error(f"  ✗ API 서버 연결 실패: {API_BASE} ({e})")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    t_start = time.time()
    n_toxic = 0
    reason_counts = {"too_short": 0, "pos_skip": 0, "hallucinated_span": 0,
                     "refused_final": 0, "parse_error": 0}

    pbar = tqdm(total=len(todo), desc="Pseudo Labeling (async)")
    processed = 0

    with open_jsonl_write(OUT_PATH, "at") as fout:
        # FLUSH_EVERY 단위로 묶어서 처리 (메모리 + 중간저장 균형)
        for chunk_start in range(0, len(todo), FLUSH_EVERY):
            chunk = todo[chunk_start:chunk_start + FLUSH_EVERY]

            # 청크 내 전부 동시 실행 (semaphore가 실제 동시성 제한)
            tasks = [classify_async(rec, sem) for rec in chunk]
            results = await asyncio.gather(*tasks)

            buffer = []
            for rec, llm_out in zip(chunk, results):
                reason = llm_out.get("reason", "")
                if reason in reason_counts:
                    reason_counts[reason] += 1
                compact = make_compact_record(rec, llm_out)
                if compact["is_toxic"]:
                    n_toxic += 1
                buffer.append(json.dumps(compact, ensure_ascii=False))

            fout.write("\n".join(buffer) + "\n")
            fout.flush()
            processed += len(chunk)
            pbar.update(len(chunk))

            elapsed = time.time() - t_start
            speed = processed / max(elapsed, 0.01)
            remain = (len(todo) - processed) / max(speed, 0.01) / 60
            pbar.set_postfix(speed=f"{speed:.1f}/s", eta=f"{remain:.0f}min")

    pbar.close()
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
        full_time_h = 5000 / max(speed, 0.01) / 3600
        logger.info(f"\n  → 5000건 풀 실행 예상: {full_time_h:.2f}시간 ({speed:.1f}건/초)")

    # 전체 통계
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


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
