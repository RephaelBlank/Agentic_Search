def load_csv(file_path: str) -> list[dict]:
    """Load data from a CSV file and return a list of dictionaries."""
    import csv
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_json(data: dict, file_path: str) -> None:
    """Save a dictionary to a JSON file."""
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def print_summary(results: list) -> None:
    """Print a summary of results."""
    from collections import defaultdict
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        key = (result["model"], result["prompt_variant"])
        counts[key]["total"] += 1
        if result.get("correct"):
            counts[key]["correct"] += 1

    print("\n── Summary ──────────────────────────────")
    for (model, variant), c in sorted(counts.items()):
        pct = 100 * c["correct"] / c["total"]
        print(f"  {model:<30} {variant:<15} {c['correct']}/{c['total']} ({pct:.0f}%)")
    print("─────────────────────────────────────────")