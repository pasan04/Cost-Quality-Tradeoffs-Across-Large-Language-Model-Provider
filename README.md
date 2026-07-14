# LLM Cost-Quality Benchmark Pipeline

A reproducible pipeline for benchmarking cost vs. quality across multiple
LLM providers on standard public tasks.

## ⚠️ API key safety — read this first

**Never put real API keys in any file in this project, in chat messages,
in commit messages, or anywhere they could be pasted/shared/committed.**
Keys belong only in your shell environment. Set them fresh each session
(or in `~/.bashrc` / `~/.zshrc` to persist):

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export TOGETHER_API_KEY="your-key-here"
```

If a key is ever exposed (pasted in chat, committed to git, posted
anywhere public), **revoke it immediately** at the provider's console and
generate a new one — treat it as compromised the moment it's visible
outside your own terminal.

## Files

| File | Purpose |
|---|---|
| `config.py` | Model registry (pricing, provider, tier) + task registry. **Edit this first.** Reads keys from env vars — never stores them. |
| `dispatcher.py` | Unified API-calling layer across Anthropic/OpenAI/Google/Together. |
| `tasks.py` | Loads public datasets (HF `datasets`) and builds standardized prompts. |
| `scorer.py` | Automatic metrics (ROUGE-L, EM, F1, pass@1) + LLM-as-judge scoring. |
| `cost_calculator.py` | Token-count → USD cost conversion. |
| `pipeline.py` | Main runner — orchestrates everything, writes results to `results/`. |
| `analyze.py` | Pareto frontier computation + cost-vs-quality plots. |

## Setup

```bash
pip install -r requirements.txt
```

Then set your keys as shown above, and open `config.py` to:
1. Fill in real `api_model_id` values for each model you want to test
   (replace every `"REPLACE_WITH_MODEL_ID"` placeholder).
2. **Verify and update pricing** (`price_in_per_million` /
   `price_out_per_million`) from each provider's current pricing page —
   the values in this file are placeholders. Update
   `PRICING_SNAPSHOT_DATE` when you do this.
3. Adjust `TASKS` sample sizes (`n_examples`) if you want a smaller/larger run.

## Usage

**Step 1 — always run a pilot first** to sanity-check the setup and estimate
real-world cost before committing to a full run:

```bash
python pipeline.py --pilot
```

This runs 5 examples per task, 1 repeat each, across all configured models.
Check `results/scored_results.csv` for errors before scaling up.

**Step 2 — estimate full-run cost** from the pilot's average token counts
(see `cost_calculator.estimate_total_cost`), then run the full benchmark:

```bash
python pipeline.py
```

Optionally restrict to specific models/tasks:

```bash
python pipeline.py --models claude-haiku,claude-sonnet --tasks classification,qa
```

**Step 3 — analyze results:**

```bash
python analyze.py --task classification --quality-metric accuracy_mean
python analyze.py --task summarization --quality-metric judge_score_mean
python analyze.py --task qa --quality-metric f1_mean
python analyze.py --task code_gen --quality-metric pass_at_1_mean
```

Each call prints a ranked "value ratio" table and saves a cost-vs-quality
scatter plot with the Pareto frontier highlighted to `results/`.

## Output files

- `results/raw_responses.jsonl` — every raw API response (for auditing/debugging)
- `results/scored_results.csv` — per-example, per-repeat scores + cost + latency
- `results/summary_by_model_task.csv` — aggregated means/std devs, ready for the paper's tables
- `results/pareto_plot.png` (per task, if you rename outputs per run)

## Important notes before publishing results

- **Code execution safety**: `scorer.run_unit_tests` executes model-generated
  code via subprocess. Run this in a sandboxed/containerized environment with
  no network access and no sensitive credentials on the machine — do not run
  untrusted model-generated code directly on a production or personal machine.
- **Judge model bias**: the LLM-as-judge model is configurable in `config.py`
  (`JUDGE_MODEL_PROVIDER` / `JUDGE_MODEL_ID`). Best practice is to use a judge
  model that is *not* among the models being evaluated, to reduce
  self-preference bias.
- **Pricing drift**: re-verify pricing immediately before your final run —
  provider prices change often. Always report the snapshot date in the paper.
- **Reproducibility**: `tasks.py` uses `random.seed(42)` for sampling — keep
  this fixed across runs so your dataset samples are reproducible.
- **Dataset IDs**: Hugging Face periodically renames/re-namespaces datasets
  and deprecates legacy script-based loading. If `pipeline.py` throws an
  `HfUriError` or similar on a dataset you haven't touched, the dataset was
  likely renamed — check `tasks.py`'s `DATASET_IDS` dict and search
  "huggingface datasets `<name>` namespace" for the current repo ID.
# Cost-Quality-Tradeoffs-Across-Large-Language-Model-Provider
