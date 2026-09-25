#!/usr/bin/env python3
"""Portable, explicit-input evaluation for GSM8K and TruthfulQA."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TRUTHFUL_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


def identity(value, revision):
    path = Path(value)
    if path.is_dir():
        if revision:
            raise ValueError("Do not specify a Hub revision for a local directory")
        config = path / "config.json"
        return {"path": str(path.resolve()), "config_sha256":
                hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None}
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub models and parent tokenizers require an immutable 40-character revision")
    return {"repo": value, "revision": revision}


def serializable(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["gsm8k", "truthfulqa"], required=True)
    p.add_argument("--model", required=True, help="Hub ID or local checkpoint directory")
    p.add_argument("--revision", help="Immutable Hub model commit")
    p.add_argument("--parent-tokenizer", required=True, help="Dense parent's Hub ID or local tokenizer directory")
    p.add_argument("--parent-revision", help="Immutable dense-parent tokenizer commit")
    p.add_argument("--output", required=True, type=Path, help="New output directory; existing directories are refused")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--parallelize", action="store_true", help="HF model sharding; use consistently within a cohort")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch size must be positive")
    model_id = identity(args.model, args.revision)
    parent_id = identity(args.parent_tokenizer, args.parent_revision)
    if args.output.exists():
        p.error("Output directory already exists; choose a new directory")

    # Heavy dependencies are intentionally imported after argument validation.
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks._yaml_loader import load_yaml
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.parent_tokenizer, revision=args.parent_revision,
                                               trust_remote_code=False)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if args.dataset == "gsm8k":
        task_files = [ROOT / "tasks/gsm8k_checked/gsm8k_checked.yaml"]
    else:
        task_root = Path(lm_eval.__file__).parent / "tasks/truthfulqa"
        task_files = [task_root / f"truthfulqa_{name}.yaml" for name in ("mc1", "mc2")]
    tasks = [load_yaml(path, resolve_func=True, recursive=True) for path in task_files]
    if args.dataset == "truthfulqa":
        for task in tasks:
            task["dataset_kwargs"] = {"revision": TRUTHFUL_REVISION}
    versions = {name: importlib.metadata.version(name) for name in
                ("lm-eval", "transformers", "torch", "datasets", "accelerate", "tokenizers")}
    protocol = {
        "id": "checked_v1", "dataset": args.dataset, "backend": "hf", "dtype": "bfloat16",
        "softmax_dtype": "float32", "batch_size": args.batch_size, "parallelize": args.parallelize,
        "context_tokens": 8192, "max_generated_tokens": 4096 if args.dataset == "gsm8k" else None,
        "num_fewshot": 4 if args.dataset == "gsm8k" else 0,
        "chat_template": args.dataset == "gsm8k", "enable_thinking": False,
        "cache": False, "seed": 0, "versions": versions, "parent_tokenizer": parent_id,
        "dataset_revision": tasks[0]["dataset_kwargs"]["revision"],
        "source_sha256": {"evaluate.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "utils.py": hashlib.sha256((ROOT / "tasks/gsm8k_checked/utils.py").read_bytes()).hexdigest()},
        "task_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in task_files},
    }
    args.output.mkdir(parents=True)
    manifest = {"status": "started", "protocol": protocol, "model": model_id,
                "quantization_config": config.to_dict().get("quantization_config"),
                "parent_identity_verified": False}

    def write_manifest():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=serializable) + "\n")

    write_manifest()
    model = HFLM(pretrained=args.model, revision=args.revision or "main", tokenizer=tokenizer,
                 dtype="bfloat16", softmax_dtype="float32", max_length=8192,
                 device=args.device, batch_size=args.batch_size, parallelize=args.parallelize,
                 trust_remote_code=False, enable_thinking=False)
    result = evaluator.simple_evaluate(
        model=model, tasks=tasks, num_fewshot=protocol["num_fewshot"],
        batch_size=args.batch_size, use_cache=None, cache_requests=False,
        apply_chat_template=protocol["chat_template"], fewshot_as_multiturn=True,
        log_samples=True, random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0,
    )
    samples = result.pop("samples")
    expected = 1319 if args.dataset == "gsm8k" else 817
    for task, rows in samples.items():
        if len(rows) != expected or len({r["doc_id"] for r in rows}) != expected:
            raise ValueError(f"{task}: incomplete or duplicate evaluation")
        with (args.output / f"samples_{task}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=serializable) + "\n")
    (args.output / "results.json").write_text(json.dumps(result, indent=2, default=serializable) + "\n")
    manifest["status"] = "complete"
    write_manifest()
    print(json.dumps(result["results"], indent=2, default=serializable))


if __name__ == "__main__":
    main()
