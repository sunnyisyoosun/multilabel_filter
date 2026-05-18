"""
classify.py (v2 — Poison Level 정책 적용)
=========================================
학습된 분류기로 새 문장의 유해성을 판단 + Poison Level(PL) 기반 차단 정책.

설계 (발표 슬라이드 기반):
  PL = 3·slang_conf + 4·cot_confidence + 3·max_category_weight    (0 ≤ PL ≤ 10)

  - slang_conf:        SlangLLM PoS 점수 기반 슬랭 밀도 (0~1)
  - cot_confidence:    분류기(MLP/LR)의 toxic 카테고리 최대 확률 (0~1)
  - max_category_weight: 탐지된 카테고리 중 가장 심각한 것의 가중치

카테고리 가중치 (위험도 기반):
  threat=1.0 | sexual_harassment=0.9 | hate_speech=0.9
  sexism=0.7 | profanity=0.7 | political=0.5 | other=0.4

Action 정책:
  PL ≥ 7      → BLOCK   (완전 차단)
  4 ≤ PL < 7  → FILTER  (유해부 마스킹)
  2 ≤ PL < 4  → WARN    (경고 + 로깅, 사용자 통과)
  PL < 2      → PASS    (안전 통과)

특수 규칙: threat 카테고리 탐지 시 PL 무관 즉시 BLOCK.
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

MODELS_DIR = Path("models")
EMBED_DIM  = 384
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "

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

# 카테고리 가중치 (위험도 기반)
CATEGORY_WEIGHTS = {
    "threat":            1.0,
    "sexual_harassment": 0.9,
    "hate_speech":       0.9,
    "sexism":            0.7,
    "profanity":         0.7,
    "political":         0.5,
    "other":             0.4,
}

# PL 공식 계수
PL_COEF_SLANG    = 3.0
PL_COEF_COT      = 4.0
PL_COEF_WEIGHT   = 3.0
PL_MAX           = 10.0

# Action 임계값
PL_BLOCK_THRESHOLD  = 7.0
PL_FILTER_THRESHOLD = 4.0
PL_WARN_THRESHOLD   = 2.0


# ──────────────────────────────────────────────
# 모델 로더 (lazy)
# ──────────────────────────────────────────────

_embed_model = None
_lr_model    = None
_mlp_model   = None


def get_embed_model(device: str = "cpu"):
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("ERROR: sentence-transformers 설치 필요", file=sys.stderr)
            sys.exit(1)
        print(f"임베딩 모델 로드 중...", file=sys.stderr)
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _embed_model


def get_lr_model():
    global _lr_model
    if _lr_model is None:
        path = MODELS_DIR / "lr_model.pkl"
        if not path.exists():
            print(f"ERROR: 모델 없음 — {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        _lr_model = bundle["model"]
    return _lr_model


def get_mlp_model():
    global _mlp_model
    if _mlp_model is None:
        import torch
        import torch.nn as nn
        path = MODELS_DIR / "mlp_model.pt"
        if not path.exists():
            print(f"ERROR: 모델 없음 — {path}", file=sys.stderr)
            sys.exit(1)

        class MLP(nn.Module):
            def __init__(self, in_dim=EMBED_DIM, hidden=(256, 128),
                         out_dim=len(CATEGORIES), dropout=0.2):
                super().__init__()
                layers = []
                prev = in_dim
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                    prev = h
                layers += [nn.Linear(prev, out_dim)]
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)

        bundle = torch.load(path, weights_only=False)
        model = MLP()
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        _mlp_model = model
    return _mlp_model


# ──────────────────────────────────────────────
# slang_conf 계산 (SlangLLM PoS)
# ──────────────────────────────────────────────

def _compute_slang_conf(text: str, lang: str = "en") -> float:
    """SlangLLM PoS 점수 기반 슬랭 밀도. 0~1"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from slang_pos_scorer import score_tokens
        scored = score_tokens(text, lang=lang, top_k=15)
        if not scored:
            return 0.0
        avg = sum(s for _, _, s in scored) / len(scored)
        return min(1.0, avg)
    except Exception:
        # Fallback: 키워드 기반
        text_lower = text.lower()
        slang_patterns = [
            r"\b(fuck|shit|bitch|damn|ass|stupid|idiot|dumb)",
            r"씨발|ㅅㅂ|씨1발|ㅆ1발|병신|ㅂㅅ|새끼|좆|지랄|ㅈㄹ",
            r"개새|미친|또라이|쓰레기|놈|년",
        ]
        hits = sum(1 for p in slang_patterns if re.search(p, text_lower))
        return min(1.0, hits * 0.3)


def _detect_lang(text: str) -> str:
    if not text:
        return "en"
    korean = len(re.findall(r"[가-힣]", text))
    total  = len(re.findall(r"\w", text))
    return "ko" if total > 0 and korean / total > 0.3 else "en"


# ──────────────────────────────────────────────
# Poison Level + Action
# ──────────────────────────────────────────────

def _compute_pl(slang_conf: float, cot_confidence: float, max_cat_weight: float) -> float:
    """PL = 3·slang + 4·cot + 3·weight, 0~10"""
    raw = (PL_COEF_SLANG  * slang_conf +
           PL_COEF_COT    * cot_confidence +
           PL_COEF_WEIGHT * max_cat_weight)
    return round(min(PL_MAX, max(0.0, raw)), 2)


def _decide_action(pl: float, predicted_cats: list) -> dict:
    """PL과 카테고리로 Action 결정. threat은 PL 무관 즉시 BLOCK."""
    if "threat" in predicted_cats:
        return {"action": "BLOCK",
                "reason": "threat 카테고리 탐지 — 안전 우선 즉시 차단",
                "icon": "[BLOCK]"}
    if pl >= PL_BLOCK_THRESHOLD:
        return {"action": "BLOCK",
                "reason": f"PL {pl:.2f} >= {PL_BLOCK_THRESHOLD} (완전 차단)",
                "icon": "[BLOCK]"}
    if pl >= PL_FILTER_THRESHOLD:
        return {"action": "FILTER",
                "reason": f"{PL_FILTER_THRESHOLD} <= PL {pl:.2f} < {PL_BLOCK_THRESHOLD} (유해부 마스킹)",
                "icon": "[FILTER]"}
    if pl >= PL_WARN_THRESHOLD:
        return {"action": "WARN",
                "reason": f"{PL_WARN_THRESHOLD} <= PL {pl:.2f} < {PL_FILTER_THRESHOLD} (경고+통과)",
                "icon": "[WARN]"}
    return {"action": "PASS",
            "reason": f"PL {pl:.2f} < {PL_WARN_THRESHOLD} (안전 통과)",
            "icon": "[PASS]"}


def _mask_text(text: str) -> str:
    """FILTER용: 유해 추정 단어 마스킹"""
    masked = text
    patterns = [
        r"\b(fuck\w*|shit\w*|bitch\w*|damn|stupid|idiot|dumb)\b",
        r"씨발|ㅅㅂ|씨1발|ㅆ1발|병신|ㅂㅅ|새끼|좆|지랄|ㅈㄹ",
        r"\b(kill|shoot|murder|hate)\w*\b",
        r"죽이|죽일|쏴|패고|잘라|패죽",
    ]
    for pat in patterns:
        masked = re.sub(pat, lambda m: "*" * len(m.group()), masked, flags=re.IGNORECASE)
    return masked


# ──────────────────────────────────────────────
# 메인 분류 함수
# ──────────────────────────────────────────────

def classify_text(text: str, model_name: str = "mlp", threshold: float = 0.5,
                  device: str = "cpu") -> dict:
    if not text or not text.strip():
        return {
            "text": text, "is_toxic": False, "predicted_categories": [],
            "all_scores": {c: 0.0 for c in CATEGORIES},
            "poison_level": {"slang_conf": 0.0, "cot_confidence": 0.0,
                             "max_category_weight": 0.0, "PL": 0.0},
            "action": {"action": "PASS", "reason": "empty", "icon": "[PASS]"},
        }

    lang = _detect_lang(text)

    # 1. 임베딩
    embedder = get_embed_model(device=device)
    emb = embedder.encode(
        [E5_PREFIX + text[:512]],
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    # 2. 분류
    if model_name == "lr":
        model = get_lr_model()
        scores = model.predict_proba(emb)[0]
    else:
        import torch
        model = get_mlp_model()
        with torch.no_grad():
            logits = model(torch.from_numpy(emb))
            scores = torch.sigmoid(logits).numpy()[0]

    all_scores = {c: float(scores[i]) for i, c in enumerate(CATEGORIES)}
    predicted = [
        {"category": c, "category_ko": CATEGORY_KO[c], "score": float(scores[i])}
        for i, c in enumerate(CATEGORIES) if scores[i] >= threshold
    ]
    predicted.sort(key=lambda x: -x["score"])
    predicted_cats = [p["category"] for p in predicted]

    # 3. PL 계산 (3 signals)
    slang_conf = _compute_slang_conf(text, lang=lang)
    cot_confidence = float(max(scores)) if len(scores) else 0.0
    max_cat_weight = max((CATEGORY_WEIGHTS[c] for c in predicted_cats), default=0.0)
    pl = _compute_pl(slang_conf, cot_confidence, max_cat_weight)

    # 4. Action
    action = _decide_action(pl, predicted_cats)

    # 5. FILTER이면 마스킹
    masked_text = _mask_text(text) if action["action"] == "FILTER" else None

    return {
        "text": text, "lang": lang,
        "is_toxic": len(predicted) > 0,
        "predicted_categories": predicted,
        "all_scores": all_scores,
        "poison_level": {
            "slang_conf":          round(slang_conf, 3),
            "cot_confidence":      round(cot_confidence, 3),
            "max_category_weight": round(max_cat_weight, 3),
            "PL":                  pl,
        },
        "action": action,
        "masked_text": masked_text,
    }


# ──────────────────────────────────────────────
# 출력 포맷
# ──────────────────────────────────────────────

def format_text(result: dict) -> str:
    text = result["text"]
    pl = result["poison_level"]
    action = result["action"]
    lines = []
    lines.append(f"\n입력: {text}")
    lines.append("─" * 70)
    lines.append(f"  {action['icon']} {action['action']}")
    lines.append(f"  사유: {action['reason']}")
    lines.append("")
    lines.append("  Poison Level 분석:")
    lines.append(f"    slang_conf      = {pl['slang_conf']:.3f}  (x {PL_COEF_SLANG})")
    lines.append(f"    cot_confidence  = {pl['cot_confidence']:.3f}  (x {PL_COEF_COT})")
    lines.append(f"    max_cat_weight  = {pl['max_category_weight']:.3f}  (x {PL_COEF_WEIGHT})")
    bar_len = int(pl["PL"] / PL_MAX * 30)
    lines.append(f"    PL = {pl['PL']:.2f} / 10.00   [{'#'*bar_len}{'-'*(30-bar_len)}]")
    if result["predicted_categories"]:
        lines.append("")
        lines.append("  탐지된 카테고리:")
        for r in result["predicted_categories"]:
            w = CATEGORY_WEIGHTS[r["category"]]
            mark = "*" if w >= 0.9 else " "
            lines.append(f"    {mark} {r['category_ko']:10s}  score={r['score']:.3f}  weight={w}")
    if result.get("masked_text"):
        lines.append("")
        lines.append(f"  마스킹 결과: {result['masked_text']}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 실행 모드
# ──────────────────────────────────────────────

def run_single(text: str, args):
    result = classify_text(text, model_name=args.model, threshold=args.threshold,
                           device=args.device)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_text(result))


def run_interactive(args):
    print("=" * 70)
    print("Toxic Speech Classifier + Poison Level (인터랙티브)")
    print(f"모델: {args.model.upper()} | 임계값: {args.threshold}")
    print("PL = 3·slang_conf + 4·cot_confidence + 3·max_cat_weight  (0~10)")
    print("Action: PL>=7 BLOCK | 4<=PL<7 FILTER | 2<=PL<4 WARN | PL<2 PASS")
    print("종료: q / exit / Ctrl+C")
    print("=" * 70)
    get_embed_model(device=args.device)
    if args.model == "lr": get_lr_model()
    else: get_mlp_model()
    print("\n준비 완료.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료."); break
        if not text: continue
        if text.lower() in ("q", "exit", "quit"):
            print("종료."); break
        result = classify_text(text, model_name=args.model, threshold=args.threshold,
                               device=args.device)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_text(result))


def run_file(args):
    in_path = Path(args.file)
    if not in_path.exists():
        print(f"ERROR: 파일 없음 — {in_path}", file=sys.stderr)
        sys.exit(1)
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"입력 {len(lines)}개 처리 중...", file=sys.stderr)
    results = [classify_text(l, model_name=args.model, threshold=args.threshold,
                             device=args.device) for l in lines]
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            import csv
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["text", "lang", "action", "PL", "slang_conf",
                            "cot_confidence", "max_cat_weight", "categories"]
                           + [f"score_{c}" for c in CATEGORIES])
                for r in results:
                    cats = ", ".join(p["category_ko"] for p in r["predicted_categories"])
                    row = [r["text"], r["lang"], r["action"]["action"],
                           r["poison_level"]["PL"],
                           r["poison_level"]["slang_conf"],
                           r["poison_level"]["cot_confidence"],
                           r["poison_level"]["max_category_weight"],
                           cats]
                    row += [f"{r['all_scores'][c]:.4f}" for c in CATEGORIES]
                    w.writerow(row)
        print(f"저장: {out_path}", file=sys.stderr)
    else:
        for r in results:
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
            else:
                print(format_text(r))


def main():
    ap = argparse.ArgumentParser(
        description="Toxic classifier + Poison Level 차단 정책",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
PL = 3·slang_conf + 4·cot_confidence + 3·max_category_weight  (0~10)
PL >= 7   BLOCK   완전 차단
4-7       FILTER  유해부 마스킹
2-4       WARN    경고 + 통과
PL < 2    PASS    안전 통과
특수: threat 탐지 시 PL 무관 즉시 BLOCK
""")
    ap.add_argument("text", nargs="?", default=None, help="분류할 문장")
    ap.add_argument("--model", choices=["lr", "mlp"], default="mlp")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.file:
        run_file(args)
    elif args.text:
        run_single(args.text, args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
