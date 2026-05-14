"""
inspect_pseudo_labels.py
=========================
시범 실행 (100건) 결과를 빠르게 검토하는 스크립트.

사용:
  python inspect_pseudo_labels.py            # 카테고리별 분포 + 샘플 30건
  python inspect_pseudo_labels.py --cat threat   # 특정 카테고리만
  python inspect_pseudo_labels.py --sample 50    # 샘플 개수 변경

확인 포인트:
  1. 유해/정상 비율이 합리적인가? (대략 30~70% 사이가 정상)
  2. 카테고리 분포가 한쪽으로 쏠려있지 않은가?
  3. 단계 실패율이 높지 않은가? (10% 미만이 정상)
  4. 샘플 라벨링이 직관적으로 맞는가? (육안 검증)
"""

import json
import gzip
import random
import argparse
from pathlib import Path
from collections import Counter

OUT_PATH = Path("data/llm_dataset/pseudo_labeled.jsonl.gz")
CATEGORIES = [
    "profanity", "hate_speech", "sexual_harassment", "sexism", "threat",
    "political", "other",
]
CATEGORY_KO = {
    "profanity": "욕설", "hate_speech":       "혐오발언", "sexual_harassment": "성희롱",
    "sexism": "성차별", "threat": "살해협박", "political": "정치",
    "other": "기타유해",
}


def load_records():
    if not OUT_PATH.exists():
        print(f"  파일 없음: {OUT_PATH}")
        return []
    opener = gzip.open if str(OUT_PATH).endswith(".gz") else open
    records = []
    with opener(OUT_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def print_distribution(records):
    n = len(records)
    n_toxic = sum(1 for r in records if r["is_toxic"])
    print(f"\n총 {n:,}건 — 유해 {n_toxic:,} ({n_toxic/n*100:.1f}%) / 정상 {n-n_toxic:,}")

    print(f"\n카테고리별:")
    for c in CATEGORIES:
        cnt = sum(1 for r in records if r["labels"].get(c) == 1)
        bar = "█" * int(cnt / max(n, 1) * 50)
        print(f"  {CATEGORY_KO[c]:8s} {cnt:5d}  {bar}")

    print(f"\n실패/특수:")
    reasons = Counter(r.get("reason", "") for r in records)
    failure_keys = ["stage1_fail", "stage2_fail", "stage3_fail",
                    "no_target_no_derog", "parse_error", "refused_final"]
    for k in failure_keys:
        if reasons.get(k):
            print(f"  {k}: {reasons[k]}건 ({reasons[k]/n*100:.1f}%)")

    print(f"\n언어별:")
    langs = Counter(r.get("lang", "?") for r in records)
    for lang, cnt in langs.most_common():
        print(f"  {lang}: {cnt}건")


def print_samples(records, target_cat=None, n=30):
    if target_cat:
        filtered = [r for r in records if r["labels"].get(target_cat) == 1]
        print(f"\n[{target_cat}] 라벨된 샘플 (총 {len(filtered)}건):")
    else:
        # 카테고리별 균형있게
        toxic = [r for r in records if r["is_toxic"]]
        clean = [r for r in records if not r["is_toxic"]]
        random.shuffle(toxic); random.shuffle(clean)
        filtered = toxic[:n*2//3] + clean[:n//3]
        print(f"\n샘플 (유해 {n*2//3} + 정상 {n//3}):")

    for r in filtered[:n]:
        active = [CATEGORY_KO[c] for c in CATEGORIES if r["labels"].get(c) == 1]
        labels_str = ",".join(active) if active else "정상"
        text = r["text"][:80]
        span = r.get("toxic_span", "")
        reason = r.get("reason", "")
        print(f"\n  [{r.get('lang','?')}] [{labels_str}]")
        print(f"     text: {text}")
        if span:
            print(f"     span: {span}")
        if reason:
            print(f"     reason: {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", help="특정 카테고리만 보기")
    ap.add_argument("--sample", type=int, default=30, help="샘플 개수")
    args = ap.parse_args()

    records = load_records()
    if not records:
        print("아직 결과 파일이 없습니다. llm_pseudo_label_v3.py를 먼저 실행하세요.")
        return

    print_distribution(records)
    print_samples(records, target_cat=args.cat, n=args.sample)

    print("\n" + "=" * 60)
    print("결과 검토 체크리스트:")
    print("  □ 유해 비율이 30~70% 범위인가?")
    print("  □ 한 카테고리(보통 other)에 80%↑ 쏠리지 않았나?")
    print("  □ 단계 실패율이 10% 미만인가?")
    print("  □ 샘플 라벨이 직관적으로 맞나?")
    print("  □ 한국어/영어 모두 합리적으로 처리되나?")
    print("=" * 60)
    print("\n괜찮으면 llm_pseudo_label_v3.py에서 MAX_SAMPLES = 5000으로 변경 후 재실행.")


if __name__ == "__main__":
    main()
