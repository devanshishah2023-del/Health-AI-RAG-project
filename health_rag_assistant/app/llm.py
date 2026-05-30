"""
Answer generation.

Two generation paths:

1. ``OpenRouterGenerator`` — calls OpenRouter's OpenAI-compatible chat-completions
   endpoint with a strict, grounding-focused prompt. Free model IDs end in
   ``:free``; the default ``openrouter/free`` is OpenRouter's auto-router, which is
   the most resilient choice because individual free models are rotated and
   retired without notice.

2. ``extractive_fallback`` — when no API key is configured (or as a safety net),
   the system returns a deterministic answer assembled directly from the
   retrieved evidence. This guarantees the answer is grounded and lets the full
   pipeline run with zero secrets, which is convenient for grading.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import httpx

from app import config
from app.chunking import Chunk

SYSTEM_PROMPT = (
    "You are a careful, plain-language health-education assistant for people "
    "living with HFpEF and cardio-kidney-metabolic conditions. Follow these rules "
    "strictly:\n"
    "1. Answer ONLY using the provided context. Do not add facts from outside the "
    "context.\n"
    "2. If the context does not contain the answer, say you do not have enough "
    "information rather than guessing.\n"
    "3. Write in clear, supportive, patient-friendly language at roughly a grade-8 "
    "reading level.\n"
    "4. Do not give a diagnosis, prescribe, or recommend changing any medication "
    "dose. Encourage the reader to talk with their own clinician.\n"
    "5. Keep the answer concise (a short paragraph or a few short bullets)."
)

PROMPT_TEMPLATE = (
    "Use only the context below to answer the patient's question.\n\n"
    "=== CONTEXT START ===\n{context}\n=== CONTEXT END ===\n\n"
    "Patient question: {question}\n\n"
    "Grounded, patient-friendly answer:"
)


def build_context(chunks: List[Chunk]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt."""
    parts = []
    for c in chunks:
        parts.append(f"[{c.document_id} | {c.chunk_id}]\n{c.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: List[Chunk]) -> str:
    return PROMPT_TEMPLATE.format(context=build_context(chunks), question=question)


def prompt_summary() -> str:
    """A short description of the prompting strategy, stored in the research log."""
    return (
        "System prompt enforces context-only answering, no diagnosis/dosing, "
        "grade-8 patient-friendly tone; user prompt injects labelled retrieved "
        "chunks as the sole context."
    )


class OpenRouterGenerator:
    """Calls OpenRouter's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        referer: str,
        title: str,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.referer = referer
        self.title = title

    def generate(self, question: str, chunks: List[Chunk]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers recommended by OpenRouter.
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def extractive_fallback(question: str, chunks: List[Chunk]) -> str:
    """
    Deterministic, fully grounded answer used when no LLM key is configured.

    It stitches together the most relevant retrieved sentences and appends a
    clear note that this is an extractive summary, so the behaviour is honest and
    never fabricates content beyond the evidence.
    """
    if not chunks:
        return (
            "I do not have enough information in my reference documents to answer "
            "that safely. Please ask your care team."
        )
    snippets = []
    for c in chunks[:2]:
        # Strip markdown heading syntax and collapse whitespace for readability.
        cleaned = re.sub(r"(?m)^#{1,6}\s*", "", c.text)
        cleaned = cleaned.replace("\n", " ").strip()
        sentences = [s.strip() for s in cleaned.split(". ") if s.strip()]
        snippet = ". ".join(sentences[:2]).strip()
        if snippet and not snippet.endswith("."):
            snippet += "."
        snippets.append(snippet)
    body = " ".join(snippets)
    return (
        f"{body}\n\n(This is an extractive summary drawn directly from the "
        "reference documents. Please confirm with your clinician.)"
    )


def build_generator():
    """Return a configured OpenRouter generator, or None to use the fallback."""
    if config.OPENROUTER_API_KEY:
        return OpenRouterGenerator(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT_SECONDS,
            referer=config.OPENROUTER_REFERER,
            title=config.OPENROUTER_TITLE,
        )
    return None


def generate_answer(question: str, chunks: List[Chunk]) -> Tuple[str, str]:
    """
    Produce an answer and report which model produced it.

    Returns (answer_text, model_name_used). On any LLM error we degrade
    gracefully to the extractive fallback so the endpoint never hard-fails.
    """
    generator = build_generator()
    if generator is not None:
        try:
            answer = generator.generate(question, chunks)
            return answer, generator.model_name
        except Exception as exc:  # network/rate-limit/etc.
            if config.USE_EXTRACTIVE_FALLBACK:
                answer = extractive_fallback(question, chunks)
                return (
                    answer + f"\n\n[note: LLM call failed ({type(exc).__name__}); "
                    "served extractive fallback]",
                    "extractive-fallback",
                )
            raise
    # No key configured.
    return extractive_fallback(question, chunks), "extractive-fallback"
