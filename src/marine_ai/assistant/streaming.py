"""Pipeline di risposta in streaming per la chat dell'interfaccia web.

Volutamente diversa dalla crew CrewAI (crew.py, usata nel notebook 06 e nella bozza
MCP): li' l'obiettivo era dimostrare l'orchestrazione multi-agente, qui l'obiettivo e'
la reattivita' in un'interfaccia interattiva — tre agenti in sequenza richiedono alcuni
minuti, inaccettabile per una chat dal vivo.

L'identificazione della specie e il recupero dei dati ambientali restano deterministici
(stesse funzioni di tools.py, stessa logica), quindi il rischio di "specie inventata"
gia' risolto per la crew resta risolto anche qui. Solo la sintesi finale usa l'LLM,
in un'unica chiamata streaming.
"""

import os
from collections.abc import Generator

from openai import OpenAI

from .tools import EnvironmentalDataTool, retrieve_top_species

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "openai/gpt-oss-20b")

_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")
_environmental_tool = EnvironmentalDataTool()

SYSTEM_PROMPT = (
    "Sei un divulgatore scientifico esperto di biologia marina. Rispondi alla domanda "
    "dell'utente usando SOLO le informazioni fornite nel contesto sottostante. Se un dato "
    "non e' disponibile nel contesto, dillo esplicitamente invece di inventarlo o stimarlo. "
    "Non sostituire mai il nome della specie indicata nel contesto con un'altra specie, "
    "anche se ti sembrasse piu' specifica o familiare. Scrivi in italiano, in modo chiaro "
    "e comprensibile anche a un pubblico non specialista."
)


def stream_answer(user_query: str) -> Generator[tuple[str, str], None, None]:
    """Genera coppie (evento, dato): 'status' per aggiornamenti di avanzamento,
    'token' per ogni frammento di testo della risposta, 'done' per il testo completo
    a fine generazione (utile al frontend per un ultimo aggiornamento pulito)."""

    retrieved = retrieve_top_species(user_query)
    if retrieved is None:
        yield "status", "Nessuna specie pertinente trovata."
        yield "done", "Non ho trovato nessuna specie pertinente a questa domanda nella base di conoscenza."
        return

    yield "status", f"Specie identificata: {retrieved['canonical_label']}. Recupero dati ambientali..."
    env_text = _environmental_tool._run(retrieved["canonical_label"])

    yield "status", "Componendo la risposta..."

    user_content = (
        f"Domanda: {user_query}\n\n"
        f"Contesto — specie identificata ({retrieved['canonical_label']}):\n{retrieved['document']}\n\n"
        f"Contesto — dati ambientali strutturati:\n{env_text}"
    )

    try:
        stream = _client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield "token", delta

        yield "done", full_text

    except Exception as e:
        error_message = (
            f"Impossibile contattare l'LLM su {LM_STUDIO_BASE_URL} (modello '{LM_STUDIO_MODEL}'). "
            f"Verifica che LM Studio sia attivo e raggiungibile da questo indirizzo. Dettaglio: {e}"
        )
        yield "llm_error", error_message
        yield "done", error_message
