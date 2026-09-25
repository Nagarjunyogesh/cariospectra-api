"""Chatbot backend — talks to any OpenAI-compatible LLM (Groq by default).

Builds a guarded system prompt from the patient's profile + screening history so
the assistant answers from the user's own data, and calls the configured model.
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from .config import settings

SYSTEM_PROMPT = """You are the CarioSpectra Assistant, a friendly dental-health helper \
inside a caries (cavity) screening app.

STYLE — be warm, genuinely helpful, and well-formatted:
- Open with a friendly 1-2 sentence answer that speaks to the person directly (use their name).
- Use bullet points (each starting with "- ") to lay out the key points clearly. For a longer
  answer, group related points under short **bold headings** on their own line.
- Put the most important words or phrases in **bold**.
- Length should fit the question: keep simple answers brief, but go into genuine detail (more
  bullets, a short extra paragraph) when it truly adds value — depth is welcome. Never pad,
  waffle, or repeat yourself.
- Always stay easy to skim: structure over walls of text.

RULES:
- General information only — you are NOT a dentist and must not diagnose.
- Use the patient's profile and screening history below; if data is missing, say so briefly.
- Recommend seeing a dentist for diagnosis/treatment. For pain, swelling, fever, or heavy \
bleeding, advise prompt care.
- Dental / oral-health topics only. Warm but concise. Never prescribe medication.

PATIENT CONTEXT:
{context}"""


def _age_from_dob(dob: Optional[str]) -> Optional[int]:
    if not dob or len(dob) < 10:
        return None
    try:
        from datetime import date

        y, m, d = int(dob[0:4]), int(dob[5:7]), int(dob[8:10])
        today = date.today()
        return today.year - y - ((today.month, today.day) < (m, d))
    except (ValueError, TypeError):
        return None


def build_context(patient: Optional[dict], scans: Optional[List[dict]]) -> str:
    lines: List[str] = []
    if patient:
        age = _age_from_dob(patient.get("dob"))
        pairs = [
            ("Name", patient.get("full_name")),
            ("Age", f"{age}" if age is not None else None),
            ("Sex", patient.get("sex")),
            ("Reported symptoms", patient.get("symptoms")),
            ("Brushing", patient.get("brushing")),
            ("Tobacco use", patient.get("tobacco")),
            ("Frequent sugary intake", patient.get("sugar")),
            ("Last dental visit", patient.get("last_visit")),
        ]
        lines.append("Patient profile:")
        lines += [f"  - {k}: {v}" for k, v in pairs if v]

    if scans:
        lines.append("")
        lines.append("Recent screening results (most recent first):")
        for s in scans[:10]:
            when = (s.get("created_at") or "")[:10]
            verdict = s.get("verdict") or "—"
            count = s.get("count", 0)
            model = s.get("model") or ""
            lines.append(f"  - {when}: {verdict} ({count} region(s)) [{model}]")

    return "\n".join(lines) if lines else "No patient data is available yet."


async def chat_completion(
    messages: List[dict],
    patient: Optional[dict],
    scans: Optional[List[dict]],
) -> str:
    """Call the configured LLM and return the assistant's reply text."""
    if not settings.llm_api_key and "localhost" not in settings.llm_base_url:
        raise RuntimeError(
            "No LLM API key configured. Set LLM_API_KEY (e.g. a free Groq key) in "
            "the backend environment, or point LLM_BASE_URL at a local Ollama."
        )

    system = SYSTEM_PROMPT.format(context=build_context(patient, scans))
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": 0.4,
        "max_tokens": 900,
    }
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
