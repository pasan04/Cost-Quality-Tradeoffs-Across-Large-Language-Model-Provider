"""
dispatcher.py

A thin abstraction layer over multiple LLM provider SDKs so the rest of the
pipeline can call `dispatch(model_config, prompt)` without caring which
provider it is. Adding a new provider = adding one branch here.

Each call returns a ModelResponse with the text, token counts, and latency,
so cost and speed can be computed uniformly downstream.
"""

import time
from dataclasses import dataclass
from typing import Optional

from config import ModelConfig, get_api_key


@dataclass
class ModelResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_seconds: float
    error: Optional[str] = None


def dispatch(model: ModelConfig, prompt: str, max_tokens: int = 1024,
             temperature: float = 0.0) -> ModelResponse:
    """Routes a prompt to the correct provider SDK and normalizes the response."""
    start = time.time()
    try:
        if model.provider == "anthropic":
            text, tin, tout = _call_anthropic(model, prompt, max_tokens, temperature)
        elif model.provider == "openai":
            text, tin, tout = _call_openai(model, prompt, max_tokens, temperature)
        elif model.provider == "google":
            text, tin, tout = _call_google(model, prompt, max_tokens, temperature)
        elif model.provider == "together":
            text, tin, tout = _call_together(model, prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {model.provider}")

        latency = time.time() - start
        return ModelResponse(text=text, tokens_in=tin, tokens_out=tout,
                              latency_seconds=latency)

    except Exception as e:
        latency = time.time() - start
        return ModelResponse(text="", tokens_in=0, tokens_out=0,
                              latency_seconds=latency, error=str(e))


# ---------------------------------------------------------------------------
# Provider-specific call implementations.
# Each returns (response_text, tokens_in, tokens_out).
# ---------------------------------------------------------------------------

def _call_anthropic(model: ModelConfig, prompt: str, max_tokens: int,
                     temperature: float):
    import anthropic

    # NOTE: newer Claude models (Opus 4.7+, Sonnet 5+) no longer accept the
    # `temperature` parameter at all — sending it, even at 0, returns a 400
    # invalid_request_error ("`temperature` is deprecated for this model").
    # These models use adaptive sampling internally instead. We therefore
    # deliberately do NOT pass temperature here. This does mean Anthropic
    # calls are not forced to temperature=0 the way other providers below
    # are — worth noting as a limitation/caveat in the paper if exact
    # cross-provider determinism matters for your methodology section.
    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    response = client.messages.create(
        model=model.api_model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return text, tokens_in, tokens_out


def _call_openai(model: ModelConfig, prompt: str, max_tokens: int,
                  temperature: float):
    from openai import OpenAI

    # NOTE: newer OpenAI models (GPT-5.x family) reject `max_tokens` entirely
    # ("Unsupported parameter... Use `max_completion_tokens` instead") AND
    # reject any explicit `temperature` value other than the model's default
    # of 1 ("Unsupported value: 'temperature' does not support 0 with this
    # model"). Like Anthropic's newer models, GPT-5.6 uses its own internal
    # reasoning/sampling behavior instead of a user-set temperature. We
    # therefore omit temperature here entirely, same as the Anthropic call.
    # This means only Google and Together are still forced to temperature=0
    # — worth expanding the Limitations-section note about cross-provider
    # determinism accordingly.
    client = OpenAI(api_key=get_api_key("openai"))
    response = client.chat.completions.create(
        model=model.api_model_id,
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    return text, tokens_in, tokens_out


def _call_google(model: ModelConfig, prompt: str, max_tokens: int,
                  temperature: float):
    import google.generativeai as genai

    genai.configure(api_key=get_api_key("google"))
    gmodel = genai.GenerativeModel(model.api_model_id)
    response = gmodel.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    text = response.text
    # Gemini's usage metadata field name may vary by SDK version — check docs.
    tokens_in = response.usage_metadata.prompt_token_count
    tokens_out = response.usage_metadata.candidates_token_count
    return text, tokens_in, tokens_out


def _call_together(model: ModelConfig, prompt: str, max_tokens: int,
                    temperature: float):
    from together import Together

    client = Together(api_key=get_api_key("together"))
    response = client.chat.completions.create(
        model=model.api_model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    return text, tokens_in, tokens_out