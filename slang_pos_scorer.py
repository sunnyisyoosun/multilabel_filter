"""
slang_pos_scorer.py (v2 — Korean enhanced)
==========================================
SlangLLM 논문 (Patel & Alsobeh, 2025) Section III.B 의 PoS Scoring 구현
+ 한국어 특화 3단계 강화

논문 식 (3): pos_score(t) = W_p × S_pos(t)
  영어 (UD PoS):
    S_pos(t) = 1.0 if INTJ | 0.8 if ADJ | 0.7 if VERB
             | 0.6 if PROPN | 0.5 if NOUN | 0.2 otherwise

한국어 강화 (v2):
  3단계 결합:
    1단계 — 욕설 사전 매칭     → score = 1.0 (감탄사급)
    2단계 — 변형 패턴 매칭     → score = 0.9 (씨1발, ㅆ1발 등)
    3단계 — 한국어 PoS 가중치  → 기본 점수

  이유: KoNLPy Okt가 욕설을 일반 명사로 분류하는 한계 보완.

설치:
  pip install spacy konlpy
  python -m spacy download en_core_web_sm
  # konlpy는 java 필요 (default-jdk)
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 영어 PoS 점수 (논문 그대로)
# ──────────────────────────────────────────────
PoS_SCORES = {
    "INTJ":  1.0, "ADJ":  0.8, "VERB": 0.7,
    "PROPN": 0.6, "NOUN": 0.5,
}
DEFAULT_SCORE = 0.2


# ──────────────────────────────────────────────
# 한국어 강화: 1단계 — 욕설 사전
# ──────────────────────────────────────────────
# filter_pseudo_labels.py와 일치. 욕설/혐오/위협 어휘.
# 토큰이 이 사전과 부분 매칭되면 score = 1.0 (최고 슬랭 신호)

KO_SLANG_DICT = frozenset([
    # 욕설 (변형 포함)
    "씨발", "ㅅㅂ", "씨1발", "ㅆ1발", "ㅆㅂ", "ㅆ1ㅂ", "ㅆㅍ", "시발", "씨파", "씨봉",
    "병신", "ㅂㅅ", "ㅄ", "개새끼", "새끼", "지랄", "ㅈㄹ", "ㅈㄴ",
    "좆", "조옺", "ㅈ같", "ㅈ에", "씹", "씨1", "씨바",
    "꺼져", "닥쳐", "엿먹", "쓰레기", "쟛같", "쟤같",
    "ㅁㅊ", "미친", "또라이", "ㅉㅉ", "ㅋㅉ",
    # 혐오/차별
    "짱깨", "쪽바리", "조센", "조선족", "외노자",
    "동성애", "신천지", "한남", "한녀", "맘충", "김치녀",
    "한남충", "한녀충", "남혐", "여혐",
    "틀딱", "노친네", "노친네들", "노친", "꼰대", "할배", "할미", "늙은이",
    "헬조선", "촌놈", "촌년",
    # 위협 (Okt 분리 대응 — "패고싶다" → "패"+"고"+"싶다", "잘라버리" → "잘라"+"버리")
    "죽이", "죽일", "죽어", "죽을", "쏠", "찌르", "패", "팰", "패고", "패고싶",
    "협박", "테러", "박살", "뽀개", "조져", "잘라버리", "잘라",
    "혼내", "응징", "처단", "두들겨", "갈겨", "패죽이", "쳐죽", "뒈져",
    # 정치 비방
    "좌빨", "보수충", "수꼴", "꼴통", "쥐닭", "문재앙", "박그네", "쥐박이",
    "빨갱이", "토착왜구", "수구", "대깨문",
])


# ──────────────────────────────────────────────
# 한국어 강화: 2단계 — 변형 패턴 (정규식)
# ──────────────────────────────────────────────
# 사전에 없어도 변형 욕설을 잡는 패턴.

KO_SLANG_PATTERNS = [
    # "씨" + 숫자/공백/특수문자 + "발" 변형 (씨1발, 씨 발, 씨ㅂ발)
    re.compile(r"씨[\s\d\W]{0,2}발"),
    re.compile(r"ㅅ[\s\W]?ㅂ"),  # ㅅㅂ, ㅅ ㅂ
    re.compile(r"ㅆ[\s\d\W]{0,2}[발ㅂ]"),  # ㅆ1발, ㅆ발, ㅆㅂ
    # 병신 변형
    re.compile(r"병[\s\W]?신"),
    re.compile(r"ㅂ[\s\W]?ㅅ"),
    # 새끼 변형
    re.compile(r"개[\s\W]?새[\s\W]?[끼키]"),
    # 한국어 욕설 강세조사 (~놈/~년/~새끼 + 아/야)
    re.compile(r"[가-힣]+(놈|년|새끼)[야아]?"),
    # 죽이/죽일 등 위협 동사
    re.compile(r"죽[\s\W]?[이일여을]"),
    # 패고싶/패죽 등 폭력 동사
    re.compile(r"패[\s\W]?(고|죽|싶)"),
    re.compile(r"잘[\s\W]?라[\s\W]?(버|줘)"),
    # 미친/존나/졸라 강조어
    re.compile(r"미[\s\W]?친"),
    re.compile(r"존[\s\W]?나"),
    re.compile(r"졸[\s\W]?라"),
    # 좆/ㅈ 변형
    re.compile(r"좆|조[\s\W]?옺"),
    re.compile(r"ㅈ[\s\W]?(같|에|ㄹ|ㄴ)"),
    # 지랄 변형
    re.compile(r"지[\s\W]?랄"),
]


# ──────────────────────────────────────────────
# 한국어 강화: 3단계 — 한국어 특화 PoS 가중치
# ──────────────────────────────────────────────
# 한국어는 영어와 달리 명사로 욕하는 경우가 많아 명사 가중치 ↑
# Okt tag set: Noun, Verb, Adjective, Adverb, Determiner, Exclamation,
#              Josa, Eomi, KoreanParticle, Foreign, Number, Unknown,
#              ProperNoun, Punctuation, Hashtag, ScreenName, etc.

KO_POS_WEIGHTS = {
    "Exclamation":   1.0,  # 감탄사 (어머, 헐, 우와) — 영어 INTJ급
    "KoreanParticle":0.9,  # ㅋㅋ, ㅎㅎ, ㅠㅠ — 감정 강조
    "Adjective":     0.8,  # 형용사
    "Verb":          0.75, # 동사 (영어보다 약간 ↑ — 한국어 동사로 욕 많이 함)
    "Noun":          0.7,  # 명사 (영어 0.5 → 0.7, 한국어 슬랭은 명사 많음)
    "ProperNoun":    0.6,  # 고유명사
    "Hashtag":       0.5,
    "Adverb":        0.4,
    "Determiner":    0.3,
}


# ──────────────────────────────────────────────
# Lazy 로더
# ──────────────────────────────────────────────
_en_nlp = None
_ko_okt = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        try:
            import spacy
            _en_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            logger.info("  spaCy en_core_web_sm 로드됨")
        except (ImportError, OSError) as e:
            logger.warning(f"  spaCy 로드 실패 ({e})")
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
            logger.warning(f"  KoNLPy 로드 실패 ({e})")
            _ko_okt = False
    return _ko_okt


# ──────────────────────────────────────────────
# 한국어 강화 점수 계산 (3단계 결합)
# ──────────────────────────────────────────────

def _ko_dict_match(token: str) -> float:
    """1단계: 사전 매칭. 토큰이 사전 단어를 포함하면 1.0"""
    # 정확 매칭
    if token in KO_SLANG_DICT:
        return 1.0
    # 부분 매칭 (예: "병신아" → "병신" 포함)
    for slang in KO_SLANG_DICT:
        if len(slang) >= 2 and slang in token:
            return 1.0
    return 0.0


def _ko_pattern_match(token: str) -> float:
    """2단계: 정규식 패턴 매칭. 변형 욕설 잡기."""
    for pat in KO_SLANG_PATTERNS:
        if pat.search(token):
            return 0.9
    return 0.0


def _score_ko_token(token: str, pos: str) -> float:
    """
    한국어 토큰의 최종 점수 — 3단계 결합:
      1) 사전 매칭 → 1.0
      2) 패턴 매칭 → 0.9
      3) PoS 가중치 → 기본
    최댓값 채택.
    """
    s_dict = _ko_dict_match(token)
    s_pat  = _ko_pattern_match(token)
    s_pos  = KO_POS_WEIGHTS.get(pos, DEFAULT_SCORE)
    return max(s_dict, s_pat, s_pos)


# ──────────────────────────────────────────────
# 영어 (논문 그대로)
# ──────────────────────────────────────────────

@lru_cache(maxsize=2048)
def score_tokens_en(text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    """영어 텍스트의 토큰별 PoS 점수 → 상위 top_k개"""
    nlp = _get_en_nlp()
    if not nlp:
        return []
    try:
        doc = nlp(text[:500])
    except Exception:
        return []
    scored = []
    for tok in doc:
        if tok.is_punct or tok.is_space or len(tok.text) < 2:
            continue
        pos   = tok.pos_
        score = PoS_SCORES.get(pos, DEFAULT_SCORE)
        scored.append((tok.text, pos, score))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


# ──────────────────────────────────────────────
# 한국어 (3단계 결합)
# ──────────────────────────────────────────────

@lru_cache(maxsize=2048)
def score_tokens_ko(text: str, top_k: int = 5) -> list[tuple[str, str, float]]:
    """
    한국어 텍스트의 토큰별 점수 — 3단계 결합 + 전체 텍스트 스캔.
    
    중요: Okt가 "씨1발" 같은 변형을 "씨"+"1"+"발"로 분리하므로,
    토큰별 매칭만으로는 변형 욕설을 놓침. 따라서 전체 텍스트에서도
    사전/패턴 매칭을 한 번 더 수행.
    
    반환: [(token, source_tag, score), ...] 상위 top_k개
    """
    scored = []

    # ── 0단계: 전체 텍스트에서 사전/패턴 매칭 (Okt 분리 문제 우회) ──
    # 사전 부분 매칭
    for slang in KO_SLANG_DICT:
        if len(slang) >= 2 and slang in text:
            scored.append((slang, "FULL_DICT", 1.0))
    # 패턴 매칭
    for pat in KO_SLANG_PATTERNS:
        m = pat.search(text)
        if m:
            scored.append((m.group(), "FULL_PATTERN", 0.9))

    # ── 1~3단계: Okt 토큰별 분석 ──
    okt = _get_ko_okt()
    if okt:
        try:
            tagged = okt.pos(text[:500], norm=True, stem=False)
            for word, tag in tagged:
                if len(word) < 1:
                    continue
                if tag in ("Punctuation", "Foreign", "Number", "URL", "Email"):
                    continue
                # 1단계 사전 매칭
                s_dict = _ko_dict_match(word)
                # 2단계 패턴 매칭
                s_pat  = _ko_pattern_match(word)
                # 3단계 PoS 가중치
                s_pos  = KO_POS_WEIGHTS.get(tag, DEFAULT_SCORE)

                if s_dict > 0 and s_dict >= max(s_pat, s_pos):
                    scored.append((word, f"{tag}+DICT", s_dict))
                elif s_pat > 0 and s_pat >= s_pos:
                    scored.append((word, f"{tag}+PATTERN", s_pat))
                else:
                    scored.append((word, tag, s_pos))
        except Exception:
            pass
    else:
        # KoNLPy 없으면 fallback (어절 분리)
        scored.extend(_ko_score_no_pos(text, top_k))

    # 중복 제거 (같은 단어는 가장 높은 점수만)
    seen = {}
    for word, tag, score in scored:
        if word not in seen or score > seen[word][1]:
            seen[word] = (tag, score)
    deduped = [(w, t, s) for w, (t, s) in seen.items()]
    deduped.sort(key=lambda x: -x[2])
    return deduped[:top_k]


def _ko_score_no_pos(text: str, top_k: int) -> list[tuple[str, str, float]]:
    """KoNLPy 없을 때 fallback: 어절 단위로 사전+패턴만 검사"""
    scored = []
    # 공백/구두점으로 분리
    tokens = re.split(r"[\s\.,!?\"'\(\)\[\]]+", text)
    for tok in tokens:
        tok = tok.strip()
        if len(tok) < 1:
            continue
        s_dict = _ko_dict_match(tok)
        s_pat  = _ko_pattern_match(tok)
        if s_dict > 0:
            scored.append((tok, "DICT", s_dict))
        elif s_pat > 0:
            scored.append((tok, "PATTERN", s_pat))
        else:
            scored.append((tok, "?", DEFAULT_SCORE))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


# ──────────────────────────────────────────────
# 공통 인터페이스
# ──────────────────────────────────────────────

def score_tokens(text: str, lang: str = "en", top_k: int = 5) -> list[tuple[str, str, float]]:
    """언어 자동 분기. lang은 'en' 또는 'ko'."""
    if lang == "ko":
        return score_tokens_ko(text, top_k)
    return score_tokens_en(text, top_k)


def format_for_prompt(scored: list[tuple[str, str, float]]) -> str:
    """LLM 프롬프트에 주입할 형식"""
    if not scored:
        return ""
    parts = []
    for word, pos, score in scored:
        parts.append(f"{word}({pos}:{score:.2f})")
    return ", ".join(parts)


# ──────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        # 영어 (변화 없음)
        ("en", "That party was lit, totally dope vibes"),
        ("en", "I'll fucking kill you if you say that"),
        # 한국어 — 사전 매칭 케이스
        ("ko", "씨발 저 틀딱들 진짜 답없네"),
        ("ko", "좌빨들은 다 빨갱이야"),
        # 한국어 — 변형 욕설 (패턴 매칭)
        ("ko", "아이 ㅆ1발 부럽다"),
        ("ko", "이 ㅅ ㅂ 같은 놈아"),
        ("ko", "씨1발 진짜 짜증나"),
        # 한국어 — 명사 위주 슬랭
        ("ko", "노친네들 줘 패고싶다"),
        # 한국어 — 정상 (점수 낮아야 함)
        ("ko", "오늘 날씨 정말 좋네요"),
        ("ko", "어제 학교에서 수업을 들었어요"),
    ]
    for lang, txt in samples:
        scored = score_tokens(txt, lang=lang, top_k=5)
        max_s = max((s for _, _, s in scored), default=0.0)
        print(f"\n[{lang}] {txt}")
        print(f"   max={max_s:.2f} | {format_for_prompt(scored)}")
