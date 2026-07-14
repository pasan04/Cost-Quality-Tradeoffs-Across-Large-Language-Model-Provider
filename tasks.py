"""
tasks.py

Loads public datasets and builds standardized prompts per task.
Uses Hugging Face `datasets` for reproducibility (pip install datasets).

NOTE ON DATASET IDS: Hugging Face deprecated bare, un-namespaced dataset
names that relied on legacy loading scripts. The IDs below are the
current, correct namespaced ones:
    - cnn_dailymail    -> "abisee/cnn_dailymail"
    - ag_news          -> "fancyzhx/ag_news"
    - hotpot_qa        -> "hotpotqa/hotpot_qa"
    - openai_humaneval -> "openai/openai_humaneval"
If you hit an HfUriError or "Repository not found" again in the future,
the dataset was likely renamed again — search "huggingface datasets
<name> namespace" to find the current repo id and update DATASET_IDS below.

Each task loader returns a list of example dicts:
    {
        "id": str,
        "prompt": str,           # fully constructed prompt to send to the model
        "reference": str | list, # ground truth (for automatic metrics)
        "meta": dict             # anything extra needed for scoring (e.g. test cases for code)
    }
"""

import random
from typing import List, Dict, Any

from config import TaskConfig

random.seed(42)  # reproducibility for sampling

# Centralized so a future rename only needs to be fixed in one place.
DATASET_IDS = {
    "summarization": "abisee/cnn_dailymail",
    "classification": "fancyzhx/ag_news",
    "qa": "hotpotqa/hotpot_qa",
    "code_gen": "openai/openai_humaneval",
}


def load_examples(task: TaskConfig) -> List[Dict[str, Any]]:
    if task.name == "summarization":
        return _load_summarization(task)
    elif task.name == "classification":
        return _load_classification(task)
    elif task.name == "qa":
        return _load_qa(task)
    elif task.name == "code_gen":
        return _load_code_gen(task)
    else:
        raise ValueError(f"Unknown task: {task.name}")


def _sample(dataset, n):
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return [dataset[i] for i in indices]


def _load_summarization(task: TaskConfig) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_IDS["summarization"], "3.0.0", split=task.split)
    rows = _sample(ds, task.n_examples)

    examples = []
    for i, row in enumerate(rows):
        prompt = (
            "Summarize the following news article in 2-3 sentences. "
            "Only output the summary, no preamble.\n\n"
            f"Article:\n{row['article']}\n\nSummary:"
        )
        examples.append({
            "id": f"summ_{i}",
            "prompt": prompt,
            "reference": row["highlights"],
            "meta": {},
        })
    return examples


def _load_classification(task: TaskConfig) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_IDS["classification"], split=task.split)
    rows = _sample(ds, task.n_examples)
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    examples = []
    for i, row in enumerate(rows):
        prompt = (
            "Classify the following news headline/text into exactly one of these "
            "categories: World, Sports, Business, Sci/Tech. "
            "Respond with only the category name.\n\n"
            f"Text: {row['text']}\n\nCategory:"
        )
        examples.append({
            "id": f"cls_{i}",
            "prompt": prompt,
            "reference": label_names[row["label"]],
            "meta": {},
        })
    return examples


def _load_qa(task: TaskConfig) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_IDS["qa"], "distractor", split=task.split)
    rows = _sample(ds, task.n_examples)

    examples = []
    for i, row in enumerate(rows):
        # Concatenate supporting context paragraphs for a RAG-style QA setup
        context_titles = row["context"]["title"]
        context_sents = row["context"]["sentences"]
        context_text = "\n".join(
            f"[{t}] " + " ".join(s) for t, s in zip(context_titles, context_sents)
        )
        prompt = (
            "Answer the question using only the provided context. "
            "Respond with a short, direct answer only.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {row['question']}\n\nAnswer:"
        )
        examples.append({
            "id": f"qa_{i}",
            "prompt": prompt,
            "reference": row["answer"],
            "meta": {},
        })
    return examples


def _load_code_gen(task: TaskConfig) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_IDS["code_gen"], split="test")
    rows = _sample(ds, task.n_examples)

    examples = []
    for i, row in enumerate(rows):
        prompt = (
            "Complete the following Python function. "
            "Only output valid Python code, no explanation, no markdown fences.\n\n"
            f"{row['prompt']}"
        )
        examples.append({
            "id": f"code_{i}",
            "prompt": prompt,
            "reference": row["canonical_solution"],
            "meta": {
                "test": row["test"],
                "entry_point": row["entry_point"],
                "task_prompt": row["prompt"],
            },
        })
    return examples
