import csv
import json
import random
import argparse
import datetime
import os
import pandas as pd
from agent import run_agentic_search, run_prefetch_search
from judge import judge_single

FILTERED_CSV = "data/freshqa_filtered.csv"

MODELS = {
    "llama-3-8b": {
        "model_id": "openai/meta-llama/Meta-Llama-3-8B-Instruct:together",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": os.environ.get("HF_TOKEN"),
    },
}

PROMPT_VARIANTS = {
    "as_of": (
        "You are an advanced research assistant. Use the web_search_tool to find accurate, up-to-date answers. "
        "All answers must reflect the state of the world as of {} — use search to verify current facts. "
        "Always answer in as few words as possible — give only the key fact or name, with no preamble, explanation, or full sentences."
    ),
}

def load_questions(n: int, seed: int | None = None) -> list[dict]:
    with open(FILTERED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    rng = random.Random(seed)
    sample = rng.sample(all_rows, min(n, len(all_rows))) if n > 0 else all_rows

    questions = []
    for row in sample:
        expected = [row[f"answer_{i}"] for i in range(10) if row.get(f"answer_{i}", "").strip()]
        questions.append({
            "id": row["id"],
            "question": row["question"],
            "expected": expected,
            "effective_year": row["effective_year"],
            "num_hops": row["num_hops"],
            "fact_type": row["fact_type"],
            "false_premise": row["false_premise"],
        })
    return questions

def run_experiment(n: int, seed: int, model_label: str, variant_name: str):
    questions = load_questions(n, seed)
    total = len(questions)
    results = []

    for q in questions:
        user_question = f"[Today is {datetime.date.today().strftime('%B %d, %Y')}] {q['question']}"
        model_cfg = MODELS[model_label]
        system_prompt = PROMPT_VARIANTS[variant_name].format(datetime.date.today().strftime('%B %d, %Y'))

        output = run_prefetch_search(
            model_name=model_cfg["model_id"],
            user_question=user_question,
            system_prompt=system_prompt,
            api_base=model_cfg.get("api_base"),
            api_key=model_cfg.get("api_key"),
        )

        raw_answer = output["answer"] or ""
        answer = raw_answer  # Simplified for this example
        expected_str = " | ".join(q["expected"])
        verdict = judge_single(q["question"], expected_str, answer)

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "expected": q["expected"],
            "model": model_label,
            "prompt_variant": variant_name,
            "answer": answer,
            "raw_answer": raw_answer,
            "correct": verdict.get("correct", False),
            "judge_reasoning": verdict.get("reasoning", ""),
        })

    output_path = f"results__{model_label}__{variant_name}__{datetime.date.today()}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def run(n: int = 5, seed: int = 42, model: str = "llama-3-8b", prompt: str = "as_of", results_path: str = "results.json"):
    """
    Minimal runner: sample up to `n` questions from ../data/freshqa_filtered.csv (if present),
    generate dummy answers, and save a simple results.json that the notebook/judge can consume.
    """
    random.seed(seed)
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'freshqa_filtered.csv')
    data_path_alt = os.path.abspath(os.path.join(os.getcwd(), 'new_order', 'colab-freshqa-experiments', 'data', 'freshqa_filtered.csv'))
    questions = None
    if os.path.exists(data_path):
        questions = pd.read_csv(data_path)
    elif os.path.exists(data_path_alt):
        questions = pd.read_csv(data_path_alt)
    else:
        # fallback: create placeholder questions
        questions = pd.DataFrame([{"question_id": i, "question": f"Placeholder question {i}"} for i in range(1, n+1)])
    sampled = questions.sample(n=min(n, len(questions)), random_state=seed).reset_index(drop=True)
    results = []
    for _, row in sampled.iterrows():
        qid = int(row.get("question_id", random.randint(1, 1_000_000)))
        qtext = str(row.get("question", ""))
        # create a dummy answer so the flow can be validated
        answer = f"(dummy answer for model={model}, prompt={prompt}) {qtext[:120]}"
        results.append({
            "question_id": qid,
            "question": qtext,
            "model": model,
            "prompt": prompt,
            "answer": answer
        })
    # write results to results_path
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FreshQA evaluation pipeline")
    parser.add_argument("-n", type=int, default=5, help="Number of questions to sample (default: 5). Use -n 0 to run all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--model", type=str, required=True, help="Model to use for the experiment.")
    parser.add_argument("--variant", type=str, required=True, help="Prompt variant to use.")
    args = parser.parse_args()

    run_experiment(n=args.n, seed=args.seed, model_label=args.model, variant_name=args.variant)