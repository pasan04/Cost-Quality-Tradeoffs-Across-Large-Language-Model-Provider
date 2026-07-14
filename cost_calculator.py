"""
cost_calculator.py

Computes per-query and aggregate cost given token counts and a model's
pricing config.
"""

from config import ModelConfig


def query_cost(model: ModelConfig, tokens_in: int, tokens_out: int) -> float:
    """Returns cost in USD for a single query."""
    cost_in = (tokens_in / 1_000_000) * model.price_in_per_million
    cost_out = (tokens_out / 1_000_000) * model.price_out_per_million
    return cost_in + cost_out


def estimate_total_cost(model: ModelConfig, avg_tokens_in: float,
                         avg_tokens_out: float, n_queries: int) -> float:
    """Useful for the pilot-run cost estimate before scaling up."""
    per_query = query_cost(model, avg_tokens_in, avg_tokens_out)
    return per_query * n_queries
