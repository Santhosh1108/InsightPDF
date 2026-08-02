"""Groq-backed answer generation with streaming and graceful error handling."""
import os
from typing import Iterator, List, Optional

from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a precise support assistant that answers ONLY using the "
    "provided document context. Rules:\n"
    "1. If the answer is not in the context, say you couldn't find it in "
    "the document — never invent information.\n"
    "2. Be concise and direct; use bullet points for lists.\n"
    "3. When helpful, mention which part of the document the info comes from.\n"
)


def _get_client(api_key=None):
    key = api_key or os.environ.get("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    return Groq(api_key=key)


def _build_messages(context: str, question: str, history: Optional[List[dict]] = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # keep only the last few turns to bound token usage
        messages.extend(history[-6:])
    messages.append(
        {
            "role": "user",
            "content": f"Context from the document:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages


def generate_answer(
    context: str,
    question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> str:
    """Non-streaming answer generation (kept for backward compatibility)."""
    client = _get_client(api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=_build_messages(context, question, history),
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Sorry, I couldn't generate an answer: {e}"


def stream_answer(
    context: str,
    question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> Iterator[str]:
    """Streaming generator — yields text chunks as they arrive."""
    client = _get_client(api_key)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=_build_messages(context, question, history),
            temperature=temperature,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"⚠️ Sorry, I couldn't generate an answer: {e}"
    
