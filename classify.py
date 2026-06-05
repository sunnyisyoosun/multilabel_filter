"""
classify.py (v5 — LLM 메인 + MLP 보조 + PL 정책)
================================================
정석 파이프라인: SlangLLM 사전차단 → HateCoT LLM 분류 → MLP 보조검증 → PL 정책

구조 (구조 A):
  1) SlangLLM PoS 사전 차단 (정상 텍스트 → LLM 호출 없이 PASS)
  2) HateCoT LLM 분류 (gemma-4-31B-it, 5단계 in-prompt CoT) — 메인
  3) MLP 보조 분류 (E5 + 카테고리별 threshold) — 검증/보강
  4) 결합: LLM 결과 + MLP 일치도 표시
  5) PL 정책: slang_conf + cot_confidence + max_cat_weight

논문 통합:
  [1] Patel & Alsobeh 2024 (SlangLLM) — PoS 사전차단
  [2] Vishwamitra et al. IEEE S&P 2024 (HateGuard/HateCoT) — 5단계 CoT
  [3] Ghorbanpour et al. 2025 (TUM) — 다국어 프롬프트

PL 공식: PL = 3·slang_conf + 4·cot_confidence + 3·max_category_weight   (0 ≤ PL ≤ 10)
Action:  PL≥7 BLOCK | 4≤PL<7 FILTER | 2≤PL<4 WARN | PL<2 PASS  + threat 즉시 BLOCK
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

# LLM API
API_BASE  = "http://203.255.188.62:8002/v1"
API_KEY   = "advanced_ba"
LLM_MODEL = "google/gemma-4-31B-it"
LLM_TIMEOUT = 30

# E5 임베딩
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "
E5_DIM = 384

CATEGORIES = [
    "profanity", "hate_speech", "gender", "threat",
    "political", "other"
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech": "혐오발언", "gender": "성 관련",
    "threat": "살해협박", "political": "정치", "other": "기타유해"
}
CATEGORY_WEIGHTS = {
    "threat": 1.0, "hate_speech": 0.9, "gender": 0.8,
    "profanity": 0.7, "political": 0.5, "other": 0.4
}

SKIP_THRESHOLD = 0.6

PL_COEF_SLANG, PL_COEF_COT, PL_COEF_WEIGHT, PL_MAX = 3.0, 4.0, 3.0, 10.0
PL_BLOCK_THRESHOLD, PL_FILTER_THRESHOLD, PL_WARN_THRESHOLD = 7.0, 4.0, 2.0

SYSTEM_PROMPT = """You are a content moderation classifier.

Categories (multi-label, [] if benign):
- profanity: vulgar/swear words
- hate_speech: attacks based on race/ethnicity/religion/nationality/LGBTQ/disability
- gender: gender-based attacks, sexism, or explicit sexual insults/content
- threat: explicit threats of violence/harm/death
- political: attacks on politicians, parties, or political ideology groups
- other: appearance shaming, ageism, regional, or other clear toxicity not above

Rules:
- Label what is EXPLICITLY in the text. Do not infer.
- Common conversation (questions, greetings, opinions) is NOT toxic.
- Empty list [] is a valid and common answer for normal text.

Think briefly through these steps (do not output them, just use them):
  1) Is there a target (person/group/identity)?
  2) Are there derogatory or harmful words explicitly?
  3) Are those words directed at the target?
  4) Do they propose or incite hate/harm?
  5) Decision: pick categories or [].

Output JSON only:
{"labels":["..."],"toxic_span":"<=30 chars","reason":"<=50 chars"}

The toxic_span MUST be a literal substring of the text. If you cannot find toxic words, return "".
"""

USER_TEMPLATE = """Text: "{text}"
Notable tokens: {pos_hint}

Output JSON:"""

MAX_REASON_LEN = 80
MAX_SPAN_LEN   = 50

_llm_client = None
_e5_model = None
_classifier = {"lr": None, "mlp": None}
_thresholds_cache = {"lr": None, "mlp": None}


def _load_llm():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        print(f"LLM 클라이언트 초기화: {API_BASE}", file=sys.stderr)
        _llm_client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=LLM_TIMEOUT)
    return _llm_client


def _load_e5(device="cpu"):
    global _e5_model
    if _e5_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"E5 임베딩 로드 중...", file=sys.stderr)
        _e5_model = SentenceTransformer(E5_MODEL_NAME, device=device)
    return _e5_model


def _load_mlp(path: Path, in_dim: int):
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, len(CATEGORIES)),
            )
        def forward(self, x):
            return self.net(x)

    bundle = torch.load(path, weights_only=False)
    model = MLP()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _load_classifier(model_name: str):
    if _classifier[model_name] is not None:
        return _classifier[model_name]
    if model_name == "lr":
        path = MODELS_DIR / "lr_model_openAI.pkl"
        if not path.exists():
            raise FileNotFoundError(f"LR 모델 없음: {path}")
        with open(path, "rb") as f:
            _classifier["lr"] = pickle.load(f)["model"]
    elif model_name == "mlp":
        path = MODELS_DIR / "mlp_model_openAI.pt"
        if not path.exists():
            raise FileNotFoundError(f"MLP 모델 없음: {path}")
        _classifier["mlp"] = _load_mlp(path, in_dim=E5_DIM)
    return _classifier[model_name]


def _load_thresholds(model_name: str, use_tuned: bool = True) -> dict:
    if not use_tuned:
        return {c: 0.5 for c in CATEGORIES}
    if _thresholds_cache[model_name] is not None:
        return _thresholds_cache[model_name]

    candidates = [
        RESULTS_DIR / f"best_thresholds_{model_name}_openAI.json",
        RESULTS_DIR / "best_thresholds_openAI.json",
        RESULTS_DIR / "best_thresholds_openAI.json",
    ]
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        thr = {c: 0.5 for c in CATEGORIES}
    else:
        with open(chosen) as f:
            d = json.load(f)["thresholds"]
        thr = {c: float(d.get(c, 0.5)) for c in CATEGORIES}
        print(f"threshold 로드: {chosen.name}", file=sys.stderr)
    _thresholds_cache[model_name] = thr
    return thr


def _slang_score(text: str, lang: str):
    """SlangLLM PoS 점수. (scored, max, pos_hint, avg) 반환."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from slang_pos_scorer import score_tokens, format_for_prompt
        scored = score_tokens(text, lang=lang, top_k=8)
        if not scored:
            return [], 0.0, "", 0.0
        max_score = max(s for _, _, s in scored)
        avg_score = sum(s for _, _, s in scored) / len(scored)
        pos_hint = format_for_prompt(scored)
        return scored, max_score, pos_hint, avg_score
    except Exception:
        text_lower = text.lower()
        patterns = [r"\b(fuck|shit|bitch|damn|stupid|idiot|kill|rape|nigger)\b",
                    r"씨발|ㅅㅂ|병신|새끼|좆|지랄|미친|쓰레기|죽여|강간"]
        hits = sum(1 for p in patterns if re.search(p, text_lower))
        avg = min(1.0, hits * 0.3)
        return [], avg, "", avg


def _call_llm(text: str, pos_hint: str, retries: int = 2) -> dict:
    """HateCoT LLM 분류. 실패 시 빈 결과."""
    client = _load_llm()
    user_prompt = USER_TEMPLATE.format(text=text, pos_hint=pos_hint or "(none)")

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _parse_llm_output(raw)
            if parsed is not None:
                return parsed
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"LLM 호출 실패: {e}")
    return {"labels": [], "toxic_span": "", "reason": "llm_unavailable"}


def _parse_llm_output(raw: str):
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


def _verify_hallucination(text: str, out: dict) -> dict:
    """toxic_span 환각 검증."""
    span = out.get("toxic_span", "")
    if out.get("labels") and span:
        text_norm = "".join(text.split()).lower()
        span_norm = "".join(span.split()).lower()
        if span_norm and span_norm not in text_norm:
            return {"labels": [], "toxic_span": "", "reason": "hallucinated_span"}
    return out


def _classify_mlp(text: str, model_name: str, use_tuned: bool, device: str) -> dict:
    """MLP/LR 보조 분류 + 카테고리별 threshold."""
    model = _load_e5(device=device)
    emb = model.encode([E5_PREFIX + text[:512]],
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    clf = _load_classifier(model_name)
    thresholds = _load_thresholds(model_name, use_tuned=use_tuned)

    if model_name == "lr":
        scores = clf.predict_proba(emb)[0]
    else:
        import torch
        with torch.no_grad():
            logits = clf(torch.from_numpy(emb))
            scores = torch.sigmoid(logits).numpy()[0]

    all_scores = {c: float(scores[i]) for i, c in enumerate(CATEGORIES)}
    predicted = [c for i, c in enumerate(CATEGORIES)
                 if float(scores[i]) >= thresholds.get(c, 0.5)]
    return {"labels": predicted, "scores": all_scores, "thresholds": thresholds}


def _detect_lang(text: str) -> str:
    if not text: return "en"
    korean = len(re.findall(r"[가-힣]", text))
    total  = len(re.findall(r"\w", text))
    return "ko" if total > 0 and korean / total > 0.3 else "en"


def _compute_pl(s, c, w):
    raw = PL_COEF_SLANG*s + PL_COEF_COT*c + PL_COEF_WEIGHT*w
    return round(min(PL_MAX, max(0.0, raw)), 2)


def _decide_action(pl, final_cats):
    if "threat" in final_cats:
        return {"action": "BLOCK", "reason": "threat 카테고리 — 즉시 차단", "icon": "[BLOCK]"}
    if pl >= PL_BLOCK_THRESHOLD:
        return {"action": "BLOCK", "reason": f"PL {pl:.2f} >= {PL_BLOCK_THRESHOLD}", "icon": "[BLOCK]"}
    if pl >= PL_FILTER_THRESHOLD:
        return {"action": "FILTER", "reason": f"{PL_FILTER_THRESHOLD} <= PL {pl:.2f} < {PL_BLOCK_THRESHOLD}", "icon": "[FILTER]"}
    if pl >= PL_WARN_THRESHOLD:
        return {"action": "WARN", "reason": f"{PL_WARN_THRESHOLD} <= PL {pl:.2f} < {PL_FILTER_THRESHOLD}", "icon": "[WARN]"}
    return {"action": "PASS", "reason": f"PL {pl:.2f} < {PL_WARN_THRESHOLD}", "icon": "[PASS]"}


def _mask_text(text, toxic_span=""):
    """toxic_span 우선 마스킹 + 정규식 백업."""
    masked = text

    # [1] LLM이 찾은 toxic_span 우선 마스킹 (대소문자 무시, 공백 정규화)
    if toxic_span:
        # 원본에서 toxic_span 위치 찾기 (대소문자 무시)
        span_norm = toxic_span.strip()
        if span_norm:
            # 정확한 위치 매칭 (case-insensitive)
            pattern = re.escape(span_norm)
            masked = re.sub(pattern, lambda m: "*" * len(m.group()),
                            masked, flags=re.IGNORECASE)

    # [2] 정규식 백업 — LLM이 놓친 욕설/위협 단어
    patterns = [
        r"\b(fuck\w*|shit\w*|bitch\w*|asshole|bastard|cunt|stupid|idiot|moron|retard|nigga)\b",
        r"\b(kill|shoot|murder|stab|destroy)\w*\b",
        r"씨발|시발|ㅅㅂ|ㅆㅂ|병신|ㅂㅅ|새끼|ㅅㄲ|좆|ㅈ같|지랄|ㅈㄹ|미친|ㅁㅊ|또라이|쓰레기|꺼져|닥쳐|빨갱이",
        r"죽이|죽일|죽어|뒤져|쏴|패죽|찔러|때려|패고|잘라|살해",
    ]
    for pat in patterns:
        masked = re.sub(pat, lambda m: "*" * len(m.group()),
                        masked, flags=re.IGNORECASE)
    return masked


def classify_text(text: str, model_name="mlp", use_tuned=True,
                  device="cpu", llm_only=False, no_llm=False) -> dict:
    if not text or not text.strip():
        return _empty_result(text)

    lang = _detect_lang(text)
    scored, max_pos, pos_hint, avg_pos = _slang_score(text, lang)
    slang_conf = avg_pos

    # PoS 사전차단
    if max_pos < SKIP_THRESHOLD and not llm_only:
        mlp_out = _classify_mlp(text, model_name, use_tuned, device) if not no_llm else None
        if not mlp_out or not mlp_out["labels"]:
            return _build_result(
                text=text, lang=lang,
                llm_out={"labels": [], "toxic_span": "", "reason": "pos_skip"},
                mlp_out=mlp_out or {"labels": [], "scores": {c: 0.0 for c in CATEGORIES},
                                    "thresholds": {c: 0.5 for c in CATEGORIES}},
                slang_conf=slang_conf, pos_score=max_pos, early_skip=True,
            )

    # LLM 호출
    if no_llm:
        llm_out = {"labels": [], "toxic_span": "", "reason": "llm_disabled"}
    else:
        llm_out_raw = _call_llm(text, pos_hint)
        llm_out = _verify_hallucination(text, llm_out_raw)

    # MLP 호출
    if llm_only:
        mlp_out = {"labels": [], "scores": {c: 0.0 for c in CATEGORIES},
                   "thresholds": {c: 0.5 for c in CATEGORIES}}
    else:
        mlp_out = _classify_mlp(text, model_name, use_tuned, device)

    return _build_result(text=text, lang=lang, llm_out=llm_out, mlp_out=mlp_out,
                         slang_conf=slang_conf, pos_score=max_pos, early_skip=False)


def _build_result(text, lang, llm_out, mlp_out, slang_conf, pos_score, early_skip):
    llm_cats = set(llm_out.get("labels", []))
    mlp_cats = set(mlp_out.get("labels", []))
    final_cats = list(llm_cats)  # LLM 메인

    # slang_conf 보정: LLM·MLP 모두 정상이면 PoS 점수 영향 감소
    if not llm_cats and not mlp_cats:
        slang_conf = slang_conf * 0.3   # 70% 감점 (정상으로 합의)
    elif not llm_cats:
        slang_conf = slang_conf * 0.6   # 40% 감점 (LLM은 정상)

    cat_analysis = []
    for c in CATEGORIES:
        if c not in (llm_cats | mlp_cats):
            continue
        in_llm = c in llm_cats
        in_mlp = c in mlp_cats
        if in_llm and in_mlp:
            agreement = "agree"
        elif in_llm:
            agreement = "llm_only"
        else:
            agreement = "mlp_only"
        cat_analysis.append({
            "category":      c,
            "category_ko":   CATEGORY_KO[c],
            "in_llm":        in_llm,
            "in_mlp":        in_mlp,
            "agreement":     agreement,
            "mlp_score":     mlp_out["scores"].get(c, 0.0),
            "mlp_threshold": mlp_out["thresholds"].get(c, 0.5),
            "weight":        CATEGORY_WEIGHTS[c],
        })

    # cot_confidence: LLM 카테고리의 MLP 평균 점수 (양쪽 일치도)
    if final_cats:
        cot_confidence = float(np.mean([mlp_out["scores"].get(c, 0.5) for c in final_cats]))
    else:
        cot_confidence = 0.0  # LLM이 무해 판단 → MLP 점수 신뢰도로 쓰지 않음

    max_cat_weight = max((CATEGORY_WEIGHTS[c] for c in final_cats), default=0.0)
    pl = _compute_pl(slang_conf, cot_confidence, max_cat_weight)
    action = _decide_action(pl, final_cats)
    # FILTER뿐 아니라 BLOCK에서도 마스킹 결과 보여줌 (사용자 확인용)
    if action["action"] in ("FILTER", "BLOCK"):
        masked_text = _mask_text(text, toxic_span=llm_out.get("toxic_span", ""))
    else:
        masked_text = None

    return {
        "text": text, "lang": lang,
        "pipeline": "SlangLLM+HateCoT+MLP",
        "early_skip": early_skip,
        "pos_max_score": round(pos_score, 3),
        "llm": {
            "labels":     list(llm_cats),
            "toxic_span": llm_out.get("toxic_span", ""),
            "reason":     llm_out.get("reason", ""),
        },
        "mlp": {
            "labels":     list(mlp_cats),
            "scores":     mlp_out["scores"],
            "thresholds": mlp_out["thresholds"],
        },
        "final_categories": [{"category": c, "category_ko": CATEGORY_KO[c],
                              "weight": CATEGORY_WEIGHTS[c]} for c in final_cats],
        "category_analysis": cat_analysis,
        "is_toxic": len(final_cats) > 0,
        "poison_level": {
            "slang_conf":          round(slang_conf, 3),
            "cot_confidence":      round(cot_confidence, 3),
            "max_category_weight": round(max_cat_weight, 3),
            "PL": pl,
        },
        "action": action,
        "masked_text": masked_text,
    }


def _empty_result(text):
    return {
        "text": text, "lang": "en", "pipeline": "SlangLLM+HateCoT+MLP",
        "early_skip": True, "pos_max_score": 0.0,
        "llm": {"labels": [], "toxic_span": "", "reason": "empty"},
        "mlp": {"labels": [], "scores": {c: 0.0 for c in CATEGORIES},
                "thresholds": {c: 0.5 for c in CATEGORIES}},
        "final_categories": [], "category_analysis": [],
        "is_toxic": False,
        "poison_level": {"slang_conf": 0.0, "cot_confidence": 0.0,
                         "max_category_weight": 0.0, "PL": 0.0},
        "action": {"action": "PASS", "reason": "empty", "icon": "[PASS]"},
        "masked_text": None,
    }


def format_text(result):
    pl = result["poison_level"]
    action = result["action"]
    lines = []
    lines.append(f"\n입력: {result['text']}")
    lines.append(f"언어: {result['lang']}  |  파이프라인: {result['pipeline']}")
    if result.get("early_skip"):
        lines.append(f"  (PoS 사전차단: max={result['pos_max_score']:.2f} < {SKIP_THRESHOLD})")
    lines.append("─" * 75)
    lines.append(f"  {action['icon']} {action['action']}")
    lines.append(f"  사유: {action['reason']}")

    llm = result["llm"]
    if llm["labels"]:
        lines.append("")
        lines.append(f"  [LLM 분류] {LLM_MODEL.split('/')[-1]}")
        lines.append(f"    카테고리: {', '.join(CATEGORY_KO[c] for c in llm['labels'])}")
        if llm["toxic_span"]:
            lines.append(f"    toxic_span: \"{llm['toxic_span']}\"")
        if llm["reason"]:
            lines.append(f"    reason: {llm['reason']}")
    elif llm["reason"]:
        lines.append(f"  [LLM] {llm['reason']}")

    if result["category_analysis"]:
        lines.append("")
        lines.append("  [카테고리 일치도 분석]")
        for ca in result["category_analysis"]:
            mark = {"agree": "[✓✓]", "llm_only": "[L  ]", "mlp_only": "[  M]"}.get(ca["agreement"], "[??]")
            lines.append(f"    {mark} {ca['category_ko']:10s}  "
                         f"MLP={ca['mlp_score']:.3f} (thr={ca['mlp_threshold']:.2f})  "
                         f"weight={ca['weight']}")
        lines.append("       (✓✓=LLM·MLP 일치  L=LLM만  M=MLP만)")

    lines.append("")
    lines.append("  [Poison Level]")
    lines.append(f"    slang_conf     = {pl['slang_conf']:.3f}  × {PL_COEF_SLANG}")
    lines.append(f"    cot_confidence = {pl['cot_confidence']:.3f}  × {PL_COEF_COT}")
    lines.append(f"    max_cat_weight = {pl['max_category_weight']:.3f}  × {PL_COEF_WEIGHT}")
    bar_len = int(pl["PL"] / PL_MAX * 30)
    lines.append(f"    PL = {pl['PL']:.2f} / 10.00   [{'#'*bar_len}{'-'*(30-bar_len)}]")

    if result.get("masked_text"):
        lines.append(f"\n  마스킹: {result['masked_text']}")
    return "\n".join(lines)


def run_single(text, args):
    result = classify_text(text, model_name=args.model,
                           use_tuned=not args.no_tuned, device=args.device,
                           llm_only=args.llm_only, no_llm=args.no_llm)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_text(result))


def run_interactive(args):
    print("=" * 75)
    print("Toxic Speech Classifier (LLM 메인 + MLP 보조 + PL 정책)")
    print(f"  LLM:  {LLM_MODEL}")
    print(f"  MLP:  {args.model.upper()} + E5 임베딩")
    print(f"  threshold: {'카테고리별 튜닝' if not args.no_tuned else '0.5 고정'}")
    if args.llm_only: print("  모드: LLM만 (MLP 비활성)")
    if args.no_llm:   print("  모드: MLP만 (LLM 비활성, fallback)")
    print("  종료: q / exit / Ctrl+C")
    print("=" * 75)
    print("\n준비 완료 (모델은 첫 입력 시 로드).\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료."); break
        if not text: continue
        if text.lower() in ("q", "exit", "quit"):
            print("종료."); break
        result = classify_text(text, model_name=args.model,
                               use_tuned=not args.no_tuned, device=args.device,
                               llm_only=args.llm_only, no_llm=args.no_llm)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_text(result))


def run_file(args):
    in_path = Path(args.file)
    if not in_path.exists():
        print(f"ERROR: 파일 없음 — {in_path}", file=sys.stderr); sys.exit(1)
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"입력 {len(lines)}개 처리 중... (LLM 호출 있어 시간 소요)", file=sys.stderr)

    results = []
    for i, line in enumerate(lines):
        r = classify_text(line, model_name=args.model,
                          use_tuned=not args.no_tuned, device=args.device,
                          llm_only=args.llm_only, no_llm=args.no_llm)
        results.append(r)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(lines)}", file=sys.stderr)

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            import csv
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["text", "lang", "action", "PL",
                            "llm_categories", "mlp_categories", "agreement_count",
                            "slang_conf", "cot_confidence", "max_cat_weight",
                            "toxic_span", "reason"])
                for r in results:
                    llm_cats = ", ".join(CATEGORY_KO[c] for c in r["llm"]["labels"])
                    mlp_cats = ", ".join(CATEGORY_KO[c] for c in r["mlp"]["labels"])
                    n_agree = sum(1 for ca in r["category_analysis"] if ca["agreement"] == "agree")
                    w.writerow([r["text"], r["lang"], r["action"]["action"],
                                r["poison_level"]["PL"], llm_cats, mlp_cats, n_agree,
                                r["poison_level"]["slang_conf"],
                                r["poison_level"]["cot_confidence"],
                                r["poison_level"]["max_category_weight"],
                                r["llm"]["toxic_span"], r["llm"]["reason"]])
        print(f"저장: {out_path}", file=sys.stderr)
    else:
        for r in results:
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
            else:
                print(format_text(r))


def main():
    ap = argparse.ArgumentParser(
        description="Toxic classifier (LLM 메인 + MLP 보조 + PL 정책)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
파이프라인: SlangLLM PoS 사전차단 → HateCoT LLM 분류 → MLP 보조검증 → PL 정책

예시:
  python classify.py "Hello world"           # 단일 텍스트
  python classify.py                         # 인터랙티브
  python classify.py --file in.txt --out out.csv
  python classify.py --llm-only "..."        # LLM만 (MLP 끔)
  python classify.py --no-llm "..."          # MLP만 (LLM 다운 시 fallback)
""")
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--model", choices=["lr", "mlp"], default="mlp")
    ap.add_argument("--no-tuned", action="store_true",
                    help="MLP threshold 0.5 고정 (실험용)")
    ap.add_argument("--llm-only", action="store_true",
                    help="MLP 호출 안 함 (LLM만)")
    ap.add_argument("--no-llm", action="store_true",
                    help="LLM 호출 안 함 (MLP만, fallback)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.llm_only and args.no_llm:
        print("ERROR: --llm-only와 --no-llm은 같이 못 씀", file=sys.stderr); sys.exit(1)

    if args.file:
        run_file(args)
    elif args.text:
        run_single(args.text, args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
