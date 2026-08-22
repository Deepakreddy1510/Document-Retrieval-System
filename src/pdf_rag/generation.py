from __future__ import annotations

from groq import Groq
from google import genai
from google.genai import types


ABSTAIN_MESSAGE = "I couldn't find enough evidence in the uploaded documents to answer that."

GROUNDING_INSTRUCTIONS = f"""You are a document-grounded question answering system.

Rules:
1. Use ONLY the supplied document context. Do not answer from outside knowledge.
2. If the context does not contain enough evidence, respond exactly with:
   {ABSTAIN_MESSAGE}
3. When you do answer, cite supporting source labels such as [S1] or [S2] directly after the supported claim.
4. Never invent a source label, document name, page number, fact, number, or quotation.
5. Prefer a concise direct answer. If sources disagree, state the disagreement and cite both.
"""


class LLMGenerator:
    """Minimal provider switch for Gemini or Groq generation.

    Retrieval, reranking, context construction, and evaluation are provider-independent.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        gemini_api_key: str | None = None,
        groq_api_key: str | None = None,
    ):
        self.provider = provider.strip().lower()
        self.model = model
        self.gemini_api_key = gemini_api_key
        self.groq_api_key = groq_api_key
        self._client = None

        if self.provider not in {"gemini", "groq"}:
            raise ValueError("LLM_PROVIDER must be either 'gemini' or 'groq'.")

    @property
    def client(self):
        if self._client is not None:
            return self._client

        if self.provider == "gemini":
            if not self.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to .env before asking questions."
                )
            self._client = genai.Client(api_key=self.gemini_api_key)
            return self._client

        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env before asking questions."
            )
        self._client = Groq(api_key=self.groq_api_key)
        return self._client

    def generate(self, *, question: str, context: str) -> str:
        if not context.strip():
            return ABSTAIN_MESSAGE

        user_prompt = f"Question:\n{question}\n\nDocument context:\n{context}"

        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=GROUNDING_INSTRUCTIONS,
                    temperature=0.1,
                ),
            )
            text = response.text or ""
            return text.strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": GROUNDING_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            reasoning_format="hidden",
            reasoning_effort="none",
            temperature=0.1,
        )

       
        return (response.choices[0].message.content or "").strip()
