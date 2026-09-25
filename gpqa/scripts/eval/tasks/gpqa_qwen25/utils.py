"""GPQA helpers for Qwen2.5 Instruct-style protocol with robust answer extraction.

Keeps the Qwen2.5 / lm-eval cot_zeroshot prompt ("Let's think step by step")
and greedy 0-shot settings, but fixes answer parsing so models that say
"**Answer:** (C)" or "\\boxed{D}" (e.g. DeepSeek-R1 distill) are not scored
as 0 by the stock "The answer is ..." strict regex.
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
_THE_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-D])\)?",
    flags=re.IGNORECASE,
)
_ANSWER_COLON_RE = re.compile(
    r"(?:\*\*)?answer(?:\*\*)?\s*:?\s*\(?([A-D])\)?",
    flags=re.IGNORECASE,
)
_PAREN_LETTER_RE = re.compile(r"\(([A-D])\)")


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
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)


def _post_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def _letter_from_blob(blob: str) -> str | None:
    m = re.search(r"\b([A-D])\b", blob, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def extract_final_letter(text: str) -> str | None:
    """Prefer final-answer formats; fall back to last (A–D) after thinking."""
    if not text:
        return None
    raw = str(text)
    tail = _post_think(raw)

    # 1) last \boxed{...}
    boxed = _BOXED_RE.findall(tail) or _BOXED_RE.findall(raw)
    if boxed:
        letter = _letter_from_blob(boxed[-1])
        if letter:
            return letter

    # 2) "The answer is (C)" / "answer is C"
    matches = list(_THE_ANSWER_IS_RE.finditer(tail))
    if matches:
        return matches[-1].group(1).upper()

    # 3) "**Answer:** (C)" / "Answer: C"
    matches = list(_ANSWER_COLON_RE.finditer(tail))
    if matches:
        return matches[-1].group(1).upper()

    # 4) last "(A)"–"(D)" in the post-think / full text
    parens = _PAREN_LETTER_RE.findall(tail) or _PAREN_LETTER_RE.findall(raw)
    if parens:
        return parens[-1].upper()

    # 5) lone letter at end of post-think
    m = re.search(r"\b([A-D])\b\s*[.!?]?\s*$", tail.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def extract_flexible_letter(text: str) -> str | None:
    """Looser: last (A–D) anywhere (stock lm-eval flexible-extract spirit)."""
    if not text:
        return None
    parens = _PAREN_LETTER_RE.findall(str(text))
    if parens:
        return parens[-1].upper()
    return extract_final_letter(text)


class StrictMatchFilter(Filter):
    """Fixed 'strict' final-answer extract → '(A)'…'(D)' or '[invalid]'."""

    def apply(self, resps, docs):
        out = []
        for inst in resps:
            extracted = []
            for resp in inst:
                letter = extract_final_letter(resp)
                extracted.append(f"({letter})" if letter else "[invalid]")
            out.append(extracted)
        return out


class FlexibleExtractFilter(Filter):
    """Flexible last-choice extract → '(A)'…'(D)' or '[invalid]'."""

    def apply(self, resps, docs):
        out = []
        for inst in resps:
            extracted = []
            for resp in inst:
                letter = extract_flexible_letter(resp)
                extracted.append(f"({letter})" if letter else "[invalid]")
            out.append(extracted)
        return out
