import json
import os


def save_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(results):
    print("\n--- Evaluation Summary ---")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.2f}")
        else:
            print(f"{k:25s}: {v}")
