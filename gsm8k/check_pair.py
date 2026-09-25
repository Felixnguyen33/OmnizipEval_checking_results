#!/usr/bin/env python3
"""Rescore two checked_v1 output folders and compare their observed protocols."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scorer", ROOT / "utils.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{folder}: run is incomplete")
    if manifest["protocol"]["dataset"] != 'gsm8k':
        raise ValueError("Expected gsm8k outputs")
    result = json.loads((folder / "results.json").read_text())
    observations, scores = {}, {}
    expected_tasks = {"gsm8k_checked"}
    if set(result["results"]) != expected_tasks:
        raise ValueError("Unexpected or missing tasks")
    for task in sorted(expected_tasks):
        rows = [json.loads(line) for line in (folder / f"samples_{task}.jsonl").open()]
        expected = 1319
        if len(rows) != expected or len({r["doc_id"] for r in rows}) != expected:
            raise ValueError(f"{task}: missing or duplicate questions")
        observed, values = {}, []
        for row in rows:
            question = row["doc"]["question"]
            if question in observed:
                raise ValueError("Duplicate question")
            observed[question] = (row["doc"], row["arguments"])
            value = scorer.process_results(row["doc"], [unwrap(row["resps"])])["exact_match"]
            metric = "exact_match"
            if not math.isclose(value, row[metric], abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"{task}: per-question score mismatch")
            values.append(value)
        score = sum(values) / expected
        if not math.isclose(score, result["results"][task][metric + ",none"], abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"{task}: aggregate mismatch")
        observations[task], scores[task] = observed, score * 100
    return manifest, observations, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("variant", type=Path)
    args = p.parse_args()
    base, base_docs, base_scores = audit(args.baseline)
    variant, variant_docs, variant_scores = audit(args.variant)
    keys = set(base["protocol"]) | set(variant["protocol"])
    differences = sorted(k for k in keys if base["protocol"].get(k) != variant["protocol"].get(k))
    matched = base_docs == variant_docs
    print(json.dumps({"saved_scores_verified": True, "baseline_percent": base_scores,
                      "variant_percent": variant_scores, "protocol_differences": differences,
                      "documents_and_prompts_match": matched,
                      "variant_quantization_config": variant.get("quantization_config"),
                      "fairness_certified": False,
                      "remaining_checks": ["Verify compression parent identity from weight/provenance evidence",
                                           "Inspect generation truncation and compression calibration settings",
                                           "Confirm actual quantization bits or weight sparsity"]}, indent=2))
    raise SystemExit(0 if matched and not differences else 1)


if __name__ == "__main__":
    main()
