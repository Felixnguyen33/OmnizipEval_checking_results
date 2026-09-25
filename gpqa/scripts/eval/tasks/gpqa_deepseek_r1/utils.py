"""GPQA helpers matching DeepSeek-R1 report-style evaluation.

Protocol (DeepSeek-R1 README / arXiv:2501.12948):
  - 0-shot generative MCQ
  - max generation length 32768
  - temperature 0.6, top_p 0.95
  - N samples per question → Average Pass@1
  - prompt ends with: reason step by step + \\boxed{}
  - no system prompt
"""

from __future__ import annotations

import random
import re

import datasets

from lm_eval.api.filter import Filter


_BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*((?:[^{}]|\{[^{}]*\})+)\s*\}",
    flags=re.DOTALL,
)
_LETTER_RE = re.compile(r"\(?([A-D])\)?", flags=re.IGNORECASE)


def preprocess(text: str | None) -> str:
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Shuffle choices (same convention as lm-eval GPQA tasks)."""

    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))
        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)


def _normalize_letter(text: str) -> str | None:
    if text is None:
        return None
    text = str(text).strip()
    # Prefer last \boxed{...}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1].strip()
    # Strip think blocks if present before letter hunt on full string
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = _LETTER_RE.search(text.replace("Answer", " ").replace("answer", " "))
    if not m:
        # last-resort: lone A-D at end
        m = re.search(r"\b([A-D])\b\s*$", text.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return f"({m.group(1).upper()})"


class ExtractAnswersFilter(Filter):
    """Extract '(A)'–'(D)' from each sampled completion (keeps all N samples)."""

    def apply(self, resps, docs):
        out = []
        for inst in resps:
            extracted = []
            for resp in inst:
                letter = _normalize_letter(resp)
                extracted.append(letter if letter is not None else "[invalid]")
            out.append(extracted)
        return out


def process_results(doc, results):
    """Average Pass@1 over N sampled completions (DeepSeek report metric)."""
    gold = doc["answer"]
    preds = results[0]
    if isinstance(preds, str):
        preds = [preds]
    scores = [1.0 if p == gold else 0.0 for p in preds]
    avg = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "pass_at_1": avg,
        "exact_match_first": float(scores[0]) if scores else 0.0,
    }
