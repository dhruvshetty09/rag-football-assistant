"""
llm.py — Thin wrapper around your LLM of choice.

OPTION A — local, free, via Ollama:
  1. Install the Ollama app from https://ollama.com
  2. In a terminal: ollama pull llama3
     (alternatives: mistral, phi3, gemma2 — all free)
  3. pip install ollama

OPTION B — hosted, via OpenAI:
  pip install openai
  Set OPENAI_API_KEY in your .env file.

call_llm() always returns a plain string regardless of which backend
you use. The agent doesn't need to know which one is running.
"""

def call_llm(prompt: str, system: str = "", use_local: bool = True) -> str:
    if use_local:
        return _call_ollama(prompt, system)
    else:
        return _call_openai(prompt, system)


def _call_ollama(prompt: str, system: str) -> str:
    import ollama
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(model="phi3", messages=messages)
    return response["message"]["content"]


def _call_openai(prompt: str, system: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content
