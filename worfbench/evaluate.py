#!/usr/bin/env python3
"""WorfBench two-shot node evaluation with vLLM chat.

One model, eight tasks. Predictions resume if a task file already matches the
gold length. Node precision, recall and F1 are scored on the same GPU after
the generator is released. InterCodeSQL and Seal-Tools have no two-shot
examples in WorfBench, so those tasks are zero-shot.
"""

import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from node_eval import build_message, eval_workflow  # noqa: E402
from prompts.eval_prompt import two_shot_example  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

TASKS = [
    "alfworld",
    "intercodesql",
    "lumos",
    "os",
    "seal_tools",
    "toolalpaca",
    "toolbench",
    "webshop",
]
GOLD_ROOT = ROOT / "gold_traj"
FEW_SHOT_TASKS = set(two_shot_example)


def messages_for(example, task):
    conv = deepcopy(example["conversations"])
    return build_message(conv, task in FEW_SHOT_TASKS, task)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def make_llm(model):
    common = dict(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="auto",
    )
    try:
        return LLM(max_model_len=16384, **common)
    except Exception as exc:
        print(f"retry shorter context after: {exc}", flush=True)
        return LLM(max_model_len=8192, enforce_eager=True, **common)


def generate_task(llm, sampling, gold, task, pred_path):
    done = load_json(pred_path) if os.path.exists(pred_path) else []
    if len(done) > len(gold):
        raise RuntimeError(f"{pred_path} longer than gold")
    pending = gold[len(done):]
    if not pending:
        print(f"skip gen {task}: {len(done)} already complete", flush=True)
        return done
    print(f"gen {task}: resume {len(done)} / {len(gold)}", flush=True)
    batch = [messages_for(ex, task) for ex in pending]
    t0 = time.time()
    outputs = llm.chat(batch, sampling, use_tqdm=True)
    for ex, out in zip(pending, outputs):
        text = out.outputs[0].text if out.outputs else ""
        done.append({"query": ex, "workflow": text})
    save_json(pred_path, done)
    print(f"saved {task} n={len(done)} in {time.time() - t0:.1f}s", flush=True)
    return done


def preds_complete(tasks, model_dir):
    for task in tasks:
        gold_path = GOLD_ROOT / task / "graph_eval.json"
        pred_path = model_dir / "pred" / f"{task}.json"
        if not pred_path.exists():
            return False
        if len(load_json(pred_path)) != len(load_json(gold_path)):
            return False
    return True


def release_llm(llm):
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None) if engine is not None else None
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def install_gpu_scorer():
    import node_eval
    import sentence_transformers

    st_cls = sentence_transformers.SentenceTransformer

    class CachedGPUEncoder(st_cls):
        def __init__(self, *args, **kwargs):
            kwargs["device"] = "cuda"
            super().__init__(*args, **kwargs)
            self._cache = {}

        def encode(self, sentences, **kwargs):
            if isinstance(sentences, str):
                sentences = [sentences]
            keys = list(sentences)
            if not keys:
                dim = self.get_sentence_embedding_dimension()
                return torch.empty((0, dim), device=self.device)
            missing = [text for text in keys if text not in self._cache]
            if missing:
                embedded = super().encode(missing, convert_to_tensor=True)
                for text, vec in zip(missing, embedded):
                    self._cache[text] = vec.detach()
            return torch.stack([self._cache[text] for text in keys])

    sentence_transformers.SentenceTransformer = CachedGPUEncoder
    original = node_eval.t_eval_nodes

    def eval_nodes(pred_graph, gt_graph, sentence_model):
        pred_nodes = [n for n in pred_graph.get("nodes", []) if n not in ("START", "END")]
        gt_nodes = [n for n in gt_graph.get("nodes", []) if n not in ("START", "END")]
        if not pred_nodes or not gt_nodes:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        return original(pred_graph, gt_graph, sentence_model)

    node_eval.t_eval_nodes = eval_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--out-root", required=True, help="Directory for predictions and scores")
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    model_dir = Path(args.out_root) / args.model.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} out={model_dir}", flush=True)

    llm = None
    if preds_complete(args.tasks, model_dir):
        print("predictions complete; skip generation", flush=True)
    else:
        llm = make_llm(args.model)
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        for task in args.tasks:
            gold = load_json(GOLD_ROOT / task / "graph_eval.json")
            generate_task(llm, sampling, gold, task, model_dir / "pred" / f"{task}.json")
    release_llm(llm)
    install_gpu_scorer()
    print(f"scoring on cuda:{os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    rows = []
    for task in args.tasks:
        gold_path = str(GOLD_ROOT / task / "graph_eval.json")
        pred_path = str(model_dir / "pred" / f"{task}.json")
        eval_path = model_dir / "eval" / f"{task}_node.json"
        gold = load_json(gold_path)
        if eval_path.exists():
            metrics = load_json(eval_path)
            print(f"skip eval {task}: {metrics}", flush=True)
        else:
            eval_workflow(
                gold_path,
                pred_path,
                "sentence-transformers/all-mpnet-base-v2",
                "node",
                str(eval_path),
            )
            metrics = load_json(eval_path)
        rows.append({
            "task": task,
            **metrics,
            "n": len(gold),
            "few_shot": task in FEW_SHOT_TASKS,
        })
        save_json(model_dir / "summary.json", {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "eval_type": "node",
            "tasks": rows,
        })
    n = len(rows)
    average = {
        "precision": sum(row["precision"] for row in rows) / n,
        "recall": sum(row["recall"] for row in rows) / n,
        "f1_score": sum(row["f1_score"] for row in rows) / n,
    }
    summary = load_json(model_dir / "summary.json")
    summary["average"] = average
    save_json(model_dir / "summary.json", summary)
    print("AVERAGE", json.dumps(average), flush=True)


if __name__ == "__main__":
    main()
