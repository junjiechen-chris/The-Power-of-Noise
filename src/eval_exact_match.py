import argparse
import glob
import json
import os
import pickle
import re
from typing import Dict, List, Tuple

import pandas as pd


def find_result_files(pred_dir: str, pattern: str | None = None) -> List[str]:
    """Collect .pkl result files in a directory, sorted by trailing index if present.

    Files are expected to follow the naming pattern ..._info_{N}.pkl but we fall back to
    lexicographic sort if no index is found.
    """
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Directory not found: {pred_dir}")

    files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".pkl")]
    if pattern:
        files = [f for f in files if pattern in os.path.basename(f)]

    idx_re = re.compile(r"(\d+)\.pkl$")

    def sort_key(path: str) -> Tuple[int, str]:
        m = idx_re.search(path)
        return (int(m.group(1)) if m else -1, path)

    return sorted(files, key=sort_key)


def load_predictions_from_pickles(files: List[str]) -> List[Dict]:
    """Flatten predictions from saved pickle files into a list of {example_id, prediction, ...}.

    Each pickle file contains a list of batch dicts; each batch dict has keys like
    'example_id', 'query', 'prompt', 'generated_answer', etc., each being a list with
    the same length for that batch.
    """
    all_preds: List[Dict] = []
    total_batches = 0
    for fp in files:
        with open(fp, "rb") as f:
            batches = pickle.load(f)
        total_batches += len(batches)
        for batch in batches:
            ids = batch.get("example_id", [])
            preds = batch.get("generated_answer", [])
            queries = batch.get("query", [])
            n = len(preds)
            for i in range(n):
                ex_id = int(ids[i])
                pred = str(preds[i]).strip()
                rec = {"example_id": ex_id, "prediction": pred}
                if i < len(queries):
                    rec["question"] = queries[i]
                all_preds.append(rec)
    if not all_preds:
        raise RuntimeError("No predictions found in provided files.")
    return all_preds


def load_dataset_answers(dataset_path: str) -> Dict[int, List[str]]:
    """Load dataset JSON and return a mapping example_id -> answers (list of strings)."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    # Try standard JSON list first, fallback to JSON lines
    try:
        df = pd.read_json(dataset_path)
    except ValueError:
        df = pd.read_json(dataset_path, lines=True)
    required_cols = {"example_id", "answers"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}, got {set(df.columns)}")
    answers_map: Dict[int, List[str]] = {}
    for ex_id, answers in zip(df["example_id"], df["answers"]):
        answers_map[int(ex_id)] = list(answers)
    return answers_map


def exact_match(pred: str, golds: List[str], case_sensitive: bool = True) -> bool:
    if not case_sensitive:
        pred = pred.lower()
        golds = [g.lower() for g in golds]
    return pred in golds


def evaluate_exact_match(
    preds: List[Dict],
    answers_map: Dict[int, List[str]],
    case_sensitive: bool = True,
    dedup_strategy: str = "first",
) -> Tuple[float, int, int, List[Tuple[int, str, List[str]]]]:
    """Compute EM over predictions joined by example_id.

    - dedup_strategy: if multiple predictions exist for the same example_id, choose 'first' or 'last'.
    Returns (em, matched, total_preds_used, mismatches)
    """
    seen: Dict[int, int] = {}
    used: List[Dict] = []
    for idx, rec in enumerate(preds):
        ex_id = rec["example_id"]
        if ex_id not in answers_map:
            continue  # skip predictions without ground truth
        if ex_id in seen:
            if dedup_strategy == "first":
                continue
            elif dedup_strategy == "last":
                # mark for replacement later
                seen[ex_id] = idx
            else:
                raise ValueError("dedup_strategy must be 'first' or 'last'")
        else:
            seen[ex_id] = idx

    # materialize chosen predictions in dataset order for stability (by example_id)
    chosen: List[Tuple[int, Dict]] = sorted(((ex_id, preds[i]) for ex_id, i in seen.items()), key=lambda x: x[0])

    hits = 0
    mismatches: List[Tuple[int, str, List[str]]] = []
    for ex_id, rec in chosen:
        pred = rec["prediction"]
        golds = answers_map[ex_id]
        if exact_match(pred, golds, case_sensitive=case_sensitive):
            hits += 1
        else:
            mismatches.append((ex_id, pred, golds))

    total = len(chosen)
    em = hits / total if total else 0.0
    return em, hits, total, mismatches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Exact Match for RAG generation outputs (.pkl)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred-dir", type=str, help="Directory with .pkl prediction files")
    group.add_argument("--pred-files", type=str, nargs="+", help="Explicit list of .pkl files (supports globs)")
    p.add_argument("--dataset", type=str, default="data/test_dataset.json", help="Path to dataset JSON")
    p.add_argument("--pattern", type=str, default=None, help="Optional filename substring to filter (e.g., 'gold_at0')")
    p.add_argument("--case-sensitive", action="store_true", help="Use case-sensitive exact match (default: False)")
    p.add_argument("--dedup", choices=["first", "last"], default="first", help="If multiple preds per example_id, pick first or last")
    p.add_argument("--show", type=int, default=0, help="Show first N mismatches")
    return p.parse_args()


def expand_files(file_args: List[str]) -> List[str]:
    files: List[str] = []
    for arg in file_args:
        expanded = glob.glob(arg)
        files.extend(expanded if expanded else [arg])
    return [f for f in files if f.endswith(".pkl")]


def main() -> None:
    args = parse_args()

    if args.pred_files:
        files = expand_files(args.pred_files)
    else:
        files = find_result_files(args.pred_dir, pattern=args.pattern)

    if not files:
        raise RuntimeError("No .pkl prediction files found.")

    preds = load_predictions_from_pickles(files)
    answers_map = load_dataset_answers(args.dataset)

    em, hits, total, mismatches = evaluate_exact_match(
        preds, answers_map, case_sensitive=args.case_sensitive, dedup_strategy=args.dedup
    )

    print(f"Files: {len(files)} | Predictions used: {total} | Hits: {hits}")
    print(f"Exact Match: {em:.4f}")

    if args.show > 0:
        k = min(args.show, len(mismatches))
        for i in range(k):
            ex_id, pred, golds = mismatches[i]
            print("---")
            print(f"example_id: {ex_id}")
            print(f"pred: {pred}")
            print(f"gold: {golds}")


if __name__ == "__main__":
    main()

