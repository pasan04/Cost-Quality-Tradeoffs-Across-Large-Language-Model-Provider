"""
config.py

Central configuration for the cost-quality benchmark.
- Model registry: which models to test, their provider, and pricing.
- Task registry: which datasets/tasks to run.
- API keys are read from environment variables — NEVER hardcode keys
  anywhere in this file, in comments, in docstrings, or in any file that
  might be committed, pasted, or shared. Set them in your shell only:

    export ANTHROPIC_API_KEY="your-key-here"
    export OPENAI_API_KEY="your-key-here"
    export GOOGLE_API_KEY="your-key-here"
    export TOGETHER_API_KEY="your-key-here"

  Run those commands directly in your terminal (or add them to
  ~/.bashrc / ~/.zshrc so they persist across sessions) — never inside
  a Python file.

PRICING NOTE: prices change frequently. Fill in current $ per 1M tokens
from each provider's pricing page before running, and record the date
you captured them (used in the paper's Limitations section).
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file (if present) into os.environ
except ImportError:
    pass  # dotenv is optional — falls back to manually-exported env vars


PRICING_SNAPSHOT_DATE = "2026-07-13"  # update this whenever you refresh prices


@dataclass
class ModelConfig:
    name: str                # display name, e.g. "claude-sonnet-4-6"
    provider: str            # "anthropic" | "openai" | "google" | "together"
    tier: str                # "frontier" | "mid" | "budget" | "open-source"
    api_model_id: str        # the actual model string the SDK expects
    price_in_per_million: float   # USD per 1M input tokens
    price_out_per_million: float  # USD per 1M output tokens
    context_window: int
    notes: str = ""


# ---- Fill in / adjust this list with the models you actually want to test ----
# IMPORTANT: replace every "REPLACE_WITH_MODEL_ID" with the real model ID
# string from that provider's current docs before running anything beyond
# --pilot (the pilot will just error cleanly on these rather than charging you).
MODELS = [
    ModelConfig(
        name="claude-opus",
        provider="anthropic",
        tier="frontier",
        api_model_id="claude-opus-4-8",
        price_in_per_million=15.00,   # PLACEHOLDER - verify current price
        price_out_per_million=75.00,  # PLACEHOLDER - verify current price
        context_window=200_000,
    ),
    ModelConfig(
        name="claude-sonnet",
        provider="anthropic",
        tier="mid",
        api_model_id="claude-sonnet-5",
        price_in_per_million=3.00,    # PLACEHOLDER - verify current price
        price_out_per_million=15.00,  # PLACEHOLDER - verify current price
        context_window=200_000,
    ),
    ModelConfig(
        name="claude-haiku",
        provider="anthropic",
        tier="budget",
        api_model_id="claude-haiku-4-5-20251001",
        price_in_per_million=0.80,    # PLACEHOLDER - verify current price
        price_out_per_million=4.00,   # PLACEHOLDER - verify current price
        context_window=200_000,
    ),
    ModelConfig(
        name="gpt-frontier",
        provider="openai",
        tier="frontier",
        api_model_id="gpt-5.6-sol",   # OpenAI's frontier-tier model as of July 2026
        price_in_per_million=5.00,    # short-context standard rate, verified July 2026
        price_out_per_million=30.00,  # short-context standard rate, verified July 2026
        context_window=1_050_000,
    ),
    ModelConfig(
        name="gemini-pro",
        provider="google",
        tier="mid",
        api_model_id="gemini-3.5-flash",
        price_in_per_million=1.50,    # paid-tier standard rate, verified July 2026
        price_out_per_million=9.00,   # paid-tier standard rate (incl. thinking tokens), verified July 2026
        context_window=1_000_000,
    ),
    ModelConfig(
        name="llama-70b",
        provider="together",
        tier="open-source",
        api_model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",   # verified current Together model ID
        price_in_per_million=1.04,    # verified July 2026
        price_out_per_million=1.04,   # verified July 2026
        context_window=131_072,
    ),
]


@dataclass
class TaskConfig:
    name: str                 # "summarization" | "classification" | "qa" | "code_gen"
    dataset: str               # HF dataset name or local path
    split: str = "test"
    n_examples: int = 100      # sample size per task
    metric: str = "rouge_l"    # metric identifier used by scorer.py
    prompt_template: str = ""  # populated in tasks.py


TASKS = [
    TaskConfig(
        name="summarization",
        dataset="cnn_dailymail",
        split="test",
        n_examples=30,
        metric="rouge_l+judge",
    ),
    TaskConfig(
        name="classification",
        dataset="ag_news",
        split="test",
        n_examples=40,
        metric="accuracy",
    ),
    TaskConfig(
        name="qa",
        dataset="hotpot_qa",
        # The "distractor" config of hotpotqa/hotpot_qa only has
        # train/validation splits (no "test" split) — "validation" is
        # the correct held-out split to use here.
        split="validation",
        n_examples=30,
        metric="exact_match+f1",
    ),
    TaskConfig(
        name="code_gen",
        dataset="openai_humaneval",
        split="test",
        n_examples=30,  # HumanEval has 164 total; sample or use all
        metric="pass_at_1",
    ),
]


# Number of times to repeat each (model, example) pair to measure consistency.
# Reduced from 3 to 2 to cut cost — still yields a usable (if less precise)
# variance estimate for the consistency analysis in Section V-D of the paper.
N_REPEATS = 2

# Which model acts as the "judge" for open-ended quality scoring.
# Best practice: use a model NOT included in MODELS above, to reduce self-preference bias.
JUDGE_MODEL_PROVIDER = "anthropic"
JUDGE_MODEL_ID = "claude-sonnet-5"

# Output locations
RESULTS_DIR = "results"
RAW_RESPONSES_FILE = os.path.join(RESULTS_DIR, "raw_responses.jsonl")
SCORED_RESULTS_FILE = os.path.join(RESULTS_DIR, "scored_results.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary_by_model_task.csv")


def get_api_key(provider: str) -> Optional[str]:
    """Reads API keys from environment variables.

    Set them in your terminal before running the pipeline — never in this
    file or any other source file:

        export ANTHROPIC_API_KEY="your-key-here"
        export OPENAI_API_KEY="your-key-here"
        export GOOGLE_API_KEY="your-key-here"
        export TOGETHER_API_KEY="your-key-here"
    """
    env_var_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    key = os.environ.get(env_var_map.get(provider, ""))
    if key is None:
        raise EnvironmentError(
            f"Missing API key for provider '{provider}'. "
            f"Set the {env_var_map.get(provider)} environment variable."
        )
    return key