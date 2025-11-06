import os
import json
from collections import Counter

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOT_FILE = os.path.join(BASE_DIR, "data", "synthetic_eval_annotations_2000.jsonl")

# Analyze label distribution
def analyze_labels():
    label_counter = Counter()
    total_spans = 0

    print(f"📊 Analyzing label distribution in: {ANNOT_FILE}")

    with open(ANNOT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            spans = record.get("spans", [])
            for span in spans:
                label = span.get("label", "UNKNOWN")
                label_counter[label] += 1
                total_spans += 1

    print("\n--- PHI Label Distribution ---")
    for label, count in label_counter.most_common():
        pct = (count / total_spans) * 100 if total_spans else 0
        print(f"{label:<15} {count:>8} ({pct:5.2f}%)")

    print(f"\nTotal spans: {total_spans}")
    print(f"Unique labels: {len(label_counter)}")
    print("\n✅ Done.")

if __name__ == "__main__":
    analyze_labels()
