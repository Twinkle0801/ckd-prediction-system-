"""
Local LLM client using Ollama.

Runs completely offline.
No API key required.
"""

import ollama

MODEL_NAME = "llama3.2:3b"


def simple_completion(prompt: str, max_tokens: int = 200) -> str:
    """
    Generate a completion using the local Ollama model.
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "num_predict": max_tokens,
            },
        )

        return response["message"]["content"]

    except Exception as e:
        return f"[OLLAMA ERROR] {e}"


def completion_with_usage(prompt: str, max_tokens: int = 200) -> dict:
    """
    Same as simple_completion(), but returns token counts if available.
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "num_predict": max_tokens,
            },
        )

        return {
            "text": response["message"]["content"],
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
        }

    except Exception as e:
        return {
            "text": f"[OLLAMA ERROR] {e}",
            "input_tokens": 0,
            "output_tokens": 0,
        }