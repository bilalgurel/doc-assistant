"""Model providers. Pick one with LLM_PROVIDER in .env: "gemini" or "anthropic".

Both functions yield ("text", str) pieces while the model writes,
then exactly one ("usage", {"input_tokens": int, "output_tokens": int}).
"""
from collections.abc import Iterator

from . import config


def stream_answer(system: str, user: str, model: str, max_tokens: int) -> Iterator[tuple[str, object]]:
    provider = config.LLM_PROVIDER
    if provider == "gemini":
        yield from _gemini(system, user, model, max_tokens)
    elif provider == "anthropic":
        yield from _anthropic(system, user, model, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'anthropic'.")


def _gemini(system: str, user: str, model: str, max_tokens: int):
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    usage = {"input_tokens": 0, "output_tokens": 0}
    stream = client.models.generate_content_stream(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
    )
    for chunk in stream:
        if chunk.text:
            yield "text", chunk.text
        meta = chunk.usage_metadata
        if meta:
            # "Thinking" tokens are billed as output, so count them too.
            usage = {
                "input_tokens": meta.prompt_token_count or 0,
                "output_tokens": (meta.candidates_token_count or 0)
                + (getattr(meta, "thoughts_token_count", None) or 0),
            }
    yield "usage", usage


def _anthropic(system: str, user: str, model: str, max_tokens: int):
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for text in stream.text_stream:
            yield "text", text
        final = stream.get_final_message()
    yield "usage", {
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }
