"""
scorer.py

Computes quality scores per task type:
- summarization: ROUGE-L + LLM-as-judge (1-5 scale)
- classification: exact-match accuracy
- qa: exact match + token-level F1
- code_gen: pass@1 (execute against unit tests in a sandboxed subprocess)

LLM-as-judge mitigations implemented here:
- Uses a fixed judge model, ideally NOT among the models being evaluated.
- Uses a structured rubric + requests a single integer score for easy parsing.
"""

import re
import subprocess
import tempfile
import textwrap
from typing import Any, Dict, List

from config import JUDGE_MODEL_PROVIDER, JUDGE_MODEL_ID, ModelConfig
from dispatcher import dispatch


# ---------------------------------------------------------------------------
# Text-based metrics
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, ref_counts.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L via longest common subsequence. For rigorous results,
    consider swapping this for the `rouge-score` package instead."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_RUBRIC_TEMPLATE = """You are an impartial evaluator. Rate the RESPONSE to the
given TASK on a scale of 1-5 based on: relevance, factual accuracy relative to the
REFERENCE, and coherence. Respond with ONLY a single integer from 1 to 5, nothing else.

TASK:
{task_prompt}

REFERENCE (ground truth / gold summary):
{reference}

RESPONSE TO EVALUATE:
{response}

Score (1-5):"""


def llm_judge_score(task_prompt: str, reference: str, response: str) -> float:
    judge_model = ModelConfig(
        name="judge",
        provider=JUDGE_MODEL_PROVIDER,
        tier="judge",
        api_model_id=JUDGE_MODEL_ID,
        price_in_per_million=0.0,
        price_out_per_million=0.0,
        context_window=200_000,
    )
    prompt = JUDGE_RUBRIC_TEMPLATE.format(
        task_prompt=task_prompt, reference=reference, response=response
    )
    result = dispatch(judge_model, prompt, max_tokens=10, temperature=0.0)
    match = re.search(r"[1-5]", result.text)
    return float(match.group()) if match else 0.0


# ---------------------------------------------------------------------------
# Code execution scoring (pass@1)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Code post-processing
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    """Models frequently wrap code in markdown fences (```python ... ```)
    even when explicitly told not to. A stray fence line breaks exec() with
    a SyntaxError, which run_unit_tests silently catches as a failure —
    making every response look like a wrong answer even when the code
    itself is correct. Strip fences defensively before execution."""
    stripped = text.strip()

    # Handle ```python ... ``` or ``` ... ``` (with or without trailing newline)
    fence_pattern = r"^```(?:python)?\s*\n(.*?)\n?```\s*$"
    match = re.match(fence_pattern, stripped, re.DOTALL)
    if match:
        return match.group(1)

    # Handle a leading fence with no matching closing fence (truncated output)
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # drop the ```python or ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    return text


def run_unit_tests(generated_code: str, test_code: str, entry_point: str,
                    timeout_seconds: int = 5) -> float:
    """Executes generated code + its test suite in an isolated subprocess.
    Returns 1.0 if tests pass, 0.0 otherwise.

    SAFETY NOTE: running model-generated code is inherently risky. Use a
    sandboxed/ephemeral environment (container, restricted user, no network)
    when actually executing this at scale — do not run untrusted model-
    generated code directly on a machine with sensitive data or credentials.

    IMPLEMENTATION NOTE: this deliberately does NOT use textwrap.dedent()
    on a combined f-string template. generated_code carries its own internal
    indentation (e.g. a function body indented 4 spaces), and if it's
    embedded inside an indented template, dedent() strips the *smallest*
    common leading whitespace across the WHOLE combined string — which
    ends up being the generated code's own body indent, not the template's
    wrapper indent. That desyncs the `def` line from its body and raises
    an IndentationError on almost every response, which the except-clause
    below silently swallows as a score of 0.0 regardless of code quality.
    Concatenating at column 0 avoids this entirely.
    """
    full_script = (
        generated_code.rstrip()
        + "\n\n"
        + test_code.rstrip()
        + f"\n\ncheck({entry_point})\nprint('PASS')\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return 1.0 if "PASS" in result.stdout else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Dispatcher for task -> metric(s)
# ---------------------------------------------------------------------------

def score_example(task_name: str, example: Dict[str, Any], response_text: str) -> Dict[str, float]:
    """Returns a dict of metric_name -> score for a single (example, response) pair."""
    scores = {}

    if task_name == "summarization":
        scores["rouge_l"] = rouge_l(response_text, example["reference"])
        scores["judge_score"] = llm_judge_score(
            example["prompt"], example["reference"], response_text
        )

    elif task_name == "classification":
        scores["accuracy"] = exact_match(response_text, example["reference"])

    elif task_name == "qa":
        scores["exact_match"] = exact_match(response_text, example["reference"])
        scores["f1"] = token_f1(response_text, example["reference"])

    elif task_name == "code_gen":
        cleaned_code = strip_code_fences(response_text)
        scores["pass_at_1"] = run_unit_tests(
            cleaned_code,
            example["meta"]["test"],
            example["meta"]["entry_point"],
        )

    else:
        raise ValueError(f"Unknown task for scoring: {task_name}")

    return scores