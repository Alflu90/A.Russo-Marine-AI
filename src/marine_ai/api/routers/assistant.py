"""Endpoint per la chat con l'assistente, in streaming via Server-Sent Events (SSE).

GET invece di POST: l'API nativa del browser per SSE (EventSource) supporta solo
richieste GET, quindi la domanda dell'utente viaggia come query string. Per testi
brevi come una domanda in chat non e' un problema; per input molto lunghi andrebbe
rivista (fetch + ReadableStream), ma non e' il caso d'uso qui.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from marine_ai.assistant.streaming import stream_answer

router = APIRouter(prefix="/api/assistant", tags=["assistente"])


def _sse_format(event: str, data: str) -> str:
    """Formatta un evento SSE, gestendo correttamente anche dati multi-riga
    (ogni riga di 'data' va prefissata separatamente, per specifica SSE)."""
    data_lines = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{data_lines}\n\n"


@router.get("/ask")
def ask_assistant(query: str):
    def event_generator():
        for event, data in stream_answer(query):
            yield _sse_format(event, data)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
