"""
classify.py (v3 — 언어 라우팅 + PL 정책)
======================================
영어/한국어를 자동 판별하여 적절한 임베딩+분류기 사용.

라우팅:
  영어 → multilingual-E5-small (384-dim) → lr_model.pkl / mlp_model.pt
  한국어 → KcELECTRA-base-v2022 (768-dim) → lr_model_ko.pkl / mlp_model_ko.pt

PL 공식:
  PL = 3·slang_conf + 4·cot_confidence + 3·max_category_weight   (0 ≤ PL ≤ 10)

Action: PL≥7 BLOCK | 4≤PL<7 FILTER | 2≤PL<4 WARN | PL<2 PASS
        + threat 카테고리 즉시 BLOCK
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

E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "
E5_DIM = 384

KO_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
KO_DIM = 768
KO_MAX_LEN = 128

CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech": "혐오발언", "sexual_harassment": "성희롱",
    "sexism": "성차별", "threat": "살해협박", "political": "정치", "other": "기타유해",
}
CATEGORY_WEIGHTS = {
    "threat": 1.0, "sexual_harassment": 0.9, "hate_speech": 0.9,
    "sexism": 0.7, "profanity": 0.7, "political": 0.5, "other": 0.4,
}

PL_COEF_SLANG, PL_COEF_COT, PL_COEF_WEIGHT, PL_MAX = 3.0, 4.0, 3.0, 10.0
PL_BLOCK_THRESHOLD, PL_FILTER_THRESHOLD, PL_WARN_THRESHOLD = 7.0, 4.0, 2.0


# Lazy 로더
_e5_model = None
_ko_tokenizer = None
_ko_model = None
_lr_en = None
_mlp_en = None
_lr_ko = None
_mlp_ko = None


def _load_e5(device="cpu"):
    global _e5_model
    if _e5_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"E5 임베딩 로드 중...", file=sys.stderr)
        _e5_model = SentenceTransformer(E5_MODEL_NAME, device=device)
    return _e5_model


def _load_kcelectra(device="cpu"):
    global _ko_tokenizer, _ko_model
    if _ko_model is None:
        from transformers import AutoTokenizer, AutoModel
        print(f"KcELECTRA 로드 중...", file=sys.stderr)
        _ko_tokenizer = AutoTokenizer.from_pretrained(KO_MODEL_NAME)
        _ko_model = AutoModel.from_pretrained(KO_MODEL_NAME).to(device)
        _ko_model.eval()
    return _ko_tokenizer, _ko_model


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


def _load_classifier(lang: str, model_name: str):
    global _lr_en, _mlp_en, _lr_ko, _mlp_ko
    if lang == "en" and model_name == "lr":
        if _lr_en is None:
            path = MODELS_DIR / "lr_model_en.pkl"
            if not path.exists(): raise FileNotFoundError(f"영어 LR 없음: {path}. train_classifier_en.py 먼저 실행.")
            with open(path, "rb") as f:
                _lr_en = pickle.load(f)["model"]
        return _lr_en
    if lang == "en" and model_name == "mlp":
        if _mlp_en is None:
            _mlp_en = _load_mlp(MODELS_DIR / "mlp_model_en.pt", in_dim=E5_DIM)
        return _mlp_en
    if lang == "ko" and model_name == "lr":
        if _lr_ko is None:
            path = MODELS_DIR / "lr_model_ko.pkl"
            if not path.exists(): raise FileNotFoundError(f"한국어 LR 없음: {path}. train_classifier_ko.py 먼저 실행.")
            with open(path, "rb") as f:
                _lr_ko = pickle.load(f)["model"]
        return _lr_ko
    if lang == "ko" and model_name == "mlp":
        if _mlp_ko is None:
            _mlp_ko = _load_mlp(MODELS_DIR / "mlp_model_ko.pt", in_dim=KO_DIM)
        return _mlp_ko
    raise ValueError(f"Unknown: {lang}/{model_name}")


def _embed_en(text: str, device="cpu"):
    model = _load_e5(device=device)
    return model.encode([E5_PREFIX + text[:512]],
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def _embed_ko(text: str, device="cpu"):
    import torch
    tokenizer, model = _load_kcelectra(device=device)
    enc = tokenizer([text[:500]], padding=True, truncation=True, max_length=KO_MAX_LEN,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    hidden = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled.cpu().numpy().astype(np.float32)


def _compute_slang_conf(text: str, lang: str = "en") -> float:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from slang_pos_scorer import score_tokens
        scored = score_tokens(text, lang=lang, top_k=15)
        if not scored: return 0.0
        avg = sum(s for _, _, s in scored) / len(scored)
        return min(1.0, avg)
    except Exception:
        text_lower = text.lower()
        patterns = [r"\b(fuck|shit|bitch|damn|stupid|idiot)",
                    r"씨발|ㅅㅂ|병신|새끼|좆|지랄|미친|쓰레기"]
        hits = sum(1 for p in patterns if re.search(p, text_lower))
        return min(1.0, hits * 0.3)


def _detect_lang(text: str) -> str:
    if not text: return "en"
    korean = len(re.findall(r"[가-힣]", text))
    total  = len(re.findall(r"\w", text))
    return "ko" if total > 0 and korean / total > 0.3 else "en"


def _compute_pl(s, c, w):
    raw = PL_COEF_SLANG*s + PL_COEF_COT*c + PL_COEF_WEIGHT*w
    return round(min(PL_MAX, max(0.0, raw)), 2)


def _decide_action(pl, predicted_cats):
    if "threat" in predicted_cats:
        return {"action": "BLOCK", "reason": "threat 카테고리 — 즉시 차단", "icon": "[BLOCK]"}
    if pl >= PL_BLOCK_THRESHOLD:
        return {"action": "BLOCK", "reason": f"PL {pl:.2f} >= {PL_BLOCK_THRESHOLD}", "icon": "[BLOCK]"}
    if pl >= PL_FILTER_THRESHOLD:
        return {"action": "FILTER", "reason": f"{PL_FILTER_THRESHOLD} <= PL {pl:.2f} < {PL_BLOCK_THRESHOLD}", "icon": "[FILTER]"}
    if pl >= PL_WARN_THRESHOLD:
        return {"action": "WARN", "reason": f"{PL_WARN_THRESHOLD} <= PL {pl:.2f} < {PL_FILTER_THRESHOLD}", "icon": "[WARN]"}
    return {"action": "PASS", "reason": f"PL {pl:.2f} < {PL_WARN_THRESHOLD}", "icon": "[PASS]"}


def _mask_text(text):
    masked = text
    patterns = [r"\b(fuck\w*|shit\w*|bitch\w*|stupid|idiot)\b",
                r"씨발|ㅅㅂ|씨1발|ㅆ1발|병신|새끼|좆|지랄|ㅈㄹ",
                r"\b(kill|shoot|murder)\w*\b",
                r"죽이|죽일|쏴|패고|잘라|패죽"]
    for pat in patterns:
        masked = re.sub(pat, lambda m: "*" * len(m.group()), masked, flags=re.IGNORECASE)
    return masked


def classify_text(text: str, model_name="mlp", threshold=0.5, device="cpu") -> dict:
    if not text or not text.strip():
        return {"text": text, "lang": "en", "embedder": "none",
                "is_toxic": False, "predicted_categories": [],
                "all_scores": {c: 0.0 for c in CATEGORIES},
                "poison_level": {"slang_conf": 0.0, "cot_confidence": 0.0,
                                 "max_category_weight": 0.0, "PL": 0.0},
                "action": {"action": "PASS", "reason": "empty", "icon": "[PASS]"}}

    lang = _detect_lang(text)
    if lang == "ko":
        emb = _embed_ko(text, device=device)
        embedder = "KcELECTRA"
    else:
        emb = _embed_en(text, device=device)
        embedder = "multilingual-E5"

    try:
        clf = _load_classifier(lang, model_name)
    except FileNotFoundError as e:
        print(f"WARNING: {e} → 영어 모델 fallback", file=sys.stderr)
        emb = _embed_en(text, device=device)
        embedder = "multilingual-E5 (fallback)"
        clf = _load_classifier("en", model_name)
        lang = f"{lang}(fallback)"

    if model_name == "lr":
        scores = clf.predict_proba(emb)[0]
    else:
        import torch
        with torch.no_grad():
            logits = clf(torch.from_numpy(emb))
            scores = torch.sigmoid(logits).numpy()[0]

    all_scores = {c: float(scores[i]) for i, c in enumerate(CATEGORIES)}
    predicted = [{"category": c, "category_ko": CATEGORY_KO[c], "score": float(scores[i])}
                 for i, c in enumerate(CATEGORIES) if scores[i] >= threshold]
    predicted.sort(key=lambda x: -x["score"])
    predicted_cats = [p["category"] for p in predicted]

    slang_conf = _compute_slang_conf(text, lang=lang.replace("(fallback)", ""))
    cot_confidence = float(max(scores)) if len(scores) else 0.0
    max_cat_weight = max((CATEGORY_WEIGHTS[c] for c in predicted_cats), default=0.0)
    pl = _compute_pl(slang_conf, cot_confidence, max_cat_weight)
    action = _decide_action(pl, predicted_cats)
    masked_text = _mask_text(text) if action["action"] == "FILTER" else None

    return {
        "text": text, "lang": lang, "embedder": embedder,
        "is_toxic": len(predicted) > 0,
        "predicted_categories": predicted,
        "all_scores": all_scores,
        "poison_level": {
            "slang_conf": round(slang_conf, 3),
            "cot_confidence": round(cot_confidence, 3),
            "max_category_weight": round(max_cat_weight, 3),
            "PL": pl,
        },
        "action": action,
        "masked_text": masked_text,
    }


def format_text(result):
    pl = result["poison_level"]
    action = result["action"]
    lines = []
    lines.append(f"\n입력: {result['text']}")
    lines.append(f"언어: {result['lang']}  |  임베딩: {result['embedder']}")
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
        lines.append(f"\n  마스킹 결과: {result['masked_text']}")
    return "\n".join(lines)


def run_single(text, args):
    result = classify_text(text, model_name=args.model, threshold=args.threshold, device=args.device)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_text(result))


def run_interactive(args):
    print("=" * 70)
    print("Toxic Speech Classifier + PL 정책 (언어 라우팅)")
    print(f"모델: {args.model.upper()} | 임계값: {args.threshold}")
    print("영어 → multilingual-E5  /  한국어 → KcELECTRA")
    print("종료: q / exit / Ctrl+C")
    print("=" * 70)
    print("\n준비 완료 (모델은 첫 입력 시 로드).\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료."); break
        if not text: continue
        if text.lower() in ("q", "exit", "quit"):
            print("종료."); break
        result = classify_text(text, model_name=args.model, threshold=args.threshold, device=args.device)
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
    print(f"입력 {len(lines)}개 처리 중...", file=sys.stderr)
    results = [classify_text(l, model_name=args.model, threshold=args.threshold, device=args.device)
               for l in lines]
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            import csv
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["text", "lang", "embedder", "action", "PL", "slang_conf",
                            "cot_confidence", "max_cat_weight", "categories"]
                           + [f"score_{c}" for c in CATEGORIES])
                for r in results:
                    cats = ", ".join(p["category_ko"] for p in r["predicted_categories"])
                    row = [r["text"], r["lang"], r["embedder"], r["action"]["action"],
                           r["poison_level"]["PL"], r["poison_level"]["slang_conf"],
                           r["poison_level"]["cot_confidence"],
                           r["poison_level"]["max_category_weight"], cats]
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
        description="Toxic classifier + PL 정책 (언어 자동 라우팅)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
영어 → multilingual-E5 → lr_model.pkl / mlp_model.pt
한국어 → KcELECTRA → lr_model_ko.pkl / mlp_model_ko.pt
""")
    ap.add_argument("text", nargs="?", default=None)
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
