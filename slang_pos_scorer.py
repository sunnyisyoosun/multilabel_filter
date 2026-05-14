"""
slang_pos_scorer.py
====================
SlangLLM 논문 (Patel & Alsobeh, 2025) Section III.B 의 PoS Scoring 구현

논문 식 (3): pos_score(t) = W_p × S_pos(t)
  S_pos(t) = 1.0 if INTJ
             0.8 if ADJ
             0.7 if VERB
             0.6 if PROPN
             0.5 if NOUN
             0.2 otherwise

이유 (논문): slang은 감탄사/형용사 형태로 자주 등장 (lit, fire, dope, sus...).
여기서는 W_p = 1로 두고 S_pos만 사용 (가중치는 LLM이 흡수).

확장:
  - 영어: spaCy en_core_web_sm 사용 (UD 표준 PoS 태그)
  - 한국어: KoNLPy Okt 사용. 한국어 슬랭은 명사 비중이 높아서 매핑 다름.
    한국어 매핑 (UD 기준 변환):
      Adjective (형용사) → 0.8
      Verb (동사)        → 0.7
      Noun (명사) /
        ProperNoun (고유명사) → 0.6 (한국어 슬랭은 명사형이 많아 약간 다운)
      KoreanParticle    → 0.7 (감탄성 어미 ㅋㅋ, ㅎㅎ 등)
      otherwise         → 0.2

설치:
  pip install spacy konlpy
  python -m spacy download en_core_web_sm
  # konlpy는 java 필요 — 시스템에 default-jdk 설치돼 있어야 함
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# 논문 식 (3) — UD 표준 PoS 태그별 점수
PoS_SCORES = {
    "INTJ":  1.0,   # 감탄사
    "ADJ":   0.8,   # 형용사
    "VERB":  0.7,   # 동사
    "PROPN": 0.6,   # 고유명사
    "NOUN":  0.5,   # 명사
}
DEFAULT_SCORE = 0.2

# 한국어 (Okt 태그 → UD 비슷한 분류로 매핑)
# Okt tag set: Noun, Verb, Adjective, Adverb, Determiner, Exclamation,
#              Josa, Eomi, KoreanParticle, Foreign, Number, Unknown,
#              ProperNoun, Punctuation, Hashtag, ScreenName, Email, URL, etc.
KO_TAG_TO_SCORE = {
    "Exclamation":   1.0,  # 감탄사 (어머, 헐, 우와)
    "KoreanParticle":0.8,  # ㅋㅋ, ㅎㅎ, ㅠㅠ — 감정 표현
    "Adjective":     0.8,  # 형용사
    "Verb":          0.7,  # 동사
    "ProperNoun":    0.6,  # 고유명사
    "Noun":          0.6,  # 한국어 슬랭은 명사형이 많음 (오징어, 틀딱, 좌빨)
    "Adverb":        0.4,  # 부사
    "Hashtag":       0.5,  # 해시태그
}


# ──────────────────────────────────────────────
# Lazy-init: 모델 로딩은 첫 호출 시
# ──────────────────────────────────────────────
_en_nlp  = None
_ko_okt  = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        try:
            import spacy
            _en_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            logger.info("  spaCy en_core_web_sm 로드됨")
        except (ImportError, OSError) as e:
            logger.warning(f"  spaCy 로드 실패 ({e}) — 영어 PoS 비활성")
            _en_nlp = False
    return _en_nlp


def _get_ko_okt():
    global _ko_okt
    if _ko_okt is None:
        try:
            from konlpy.tag import Okt
            _ko_okt = Okt()
            logger.info("  KoNLPy Okt 로드됨")
        except Exception as e:
            logger.warning(f"  KoNLPy 로드 실패 ({e}) — 한국어 PoS 비활성")
            _ko_okt = False
    return _ko_okt


# ──────────────────────────────────────────────
# PoS 점수 계산
# ──────────────────────────────────────────────

@lru_cache(maxsize=2048)
def score_tokens_en(text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    """영어 텍스트의 토큰별 PoS 점수 → 상위 top_k개 (token, pos, score)"""
    nlp = _get_en_nlp()
    if not nlp:
        return []
    try:
        doc = nlp(text[:500])  # 너무 긴 텍스트 방지
    except Exception:
        return []
    scored = []
    for tok in doc:
        if tok.is_punct or tok.is_space or len(tok.text) < 2:
            continue
        pos  = tok.pos_  # spaCy는 UD pos
        score = PoS_SCORES.get(pos, DEFAULT_SCORE)
        scored.append((tok.text, pos, score))
    # 점수 높은 순으로
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


@lru_cache(maxsize=2048)
def score_tokens_ko(text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    """한국어 텍스트의 토큰별 PoS 점수 → 상위 top_k개"""
    okt = _get_ko_okt()
    if not okt:
        return []
    try:
        tagged = okt.pos(text[:500], norm=True, stem=False)
    except Exception:
        return []
    scored = []
    for word, tag in tagged:
        if len(word) < 1:
            continue
        if tag in ("Punctuation", "Foreign", "Number", "URL", "Email"):
            continue
        score = KO_TAG_TO_SCORE.get(tag, DEFAULT_SCORE)
        scored.append((word, tag, score))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


def score_tokens(text: str, lang: str = "en", top_k: int = 5) -> list[tuple[str, str, float]]:
    """언어 자동 분기. lang은 'en' 또는 'ko'."""
    if lang == "ko":
        return score_tokens_ko(text, top_k)
    return score_tokens_en(text, top_k)


def format_for_prompt(scored: list[tuple[str, str, float]]) -> str:
    """LLM 프롬프트에 주입할 형식으로 변환"""
    if not scored:
        return ""
    parts = []
    for word, pos, score in scored:
        parts.append(f"{word}({pos}:{score})")
    return ", ".join(parts)


# ──────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        ("en", "That party was lit, totally dope vibes"),
        ("en", "I'll fucking kill you if you say that"),
        ("ko", "이 영화 진짜 lit했어 ㅋㅋ"),
        ("ko", "씨발 저 틀딱들 진짜 답없네"),
        ("ko", "좌빨들은 다 빨갱이야"),
    ]
    for lang, txt in samples:
        scored = score_tokens(txt, lang=lang, top_k=5)
        print(f"\n[{lang}] {txt}")
        print(f"   → {format_for_prompt(scored)}")
