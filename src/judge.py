import json
import os
import random
import time
import litellm
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial answer evaluator. "
    "Given a question, the correct answer, and a model's answer, decide if the model's answer is correct. "
    "Be lenient with phrasing — if the meaning is the same, mark it correct. "
    "Respond ONLY with valid JSON: {\"correct\": true/false, \"reasoning\": \"one sentence\"}"
)

def judge_single(question: str, expected: str, model_answer: str) -> dict:
    user_msg = (
        f"Question: {question}\n"
        f"Correct answer: {expected}\n"
        f"Model's answer: {model_answer}"
    )
    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"correct": False, "reasoning": f"Judge error: {e}"}

def run_judge(results_path: str = "results.json", output_path: str = "judgements.json"):
    """
    Minimal judge: read results.json and assign a simple score/judgement for each entry.
    This is a placeholder so the notebook can run end-to-end.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    judgements: List[Dict] = []
    for r in results:
        score = random.choice([0, 1])  # placeholder binary score
        judgements.append({
            "question_id": r.get("question_id"),
            "model": r.get("model"),
            "prompt": r.get("prompt"),
            "score": score,
            "note": "placeholder judgement"
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgements, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(judgements)} judgements to {output_path}")
    return judgements

    judgements = []
    for i, result in enumerate(results):
        verdict = judge_single(
            question=result["question"],
            expected=result["expected"],
            model_answer=result["answer"],
        )
        time.sleep(2)
        judgements.append({
            **result,
            "correct": verdict.get("correct", False),
            "judge_reasoning": verdict.get("reasoning", ""),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judgements, f, ensure_ascii=False, indent=2)

    _print_summary(judgements)

def _print_summary(judgements: list) -> None:
    from collections import defaultdict
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for j in judgements:
        key = (j["model"], j["prompt_variant"])
        counts[key]["total"] += 1
        if j["correct"]:
            counts[key]["correct"] += 1

    print("\n── Summary ──────────────────────────────")
    for (model, variant), c in sorted(counts.items()):
        pct = 100 * c["correct"] / c["total"]
        print(f"  {model:<30} {variant:<15} {c['correct']}/{c['total']} ({pct:.0f}%)")
    print("─────────────────────────────────────────")