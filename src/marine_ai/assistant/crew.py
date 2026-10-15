"""Crew a tre agenti: tassonomo, analista ambientale, comunicatore.

Nota architetturale importante: l'identificazione della specie NON e' lasciata alla
discrezione dell'agente tassonomo. E' stato osservato empiricamente che, quando puo'
scegliere liberamente, il modello a volte ignora il risultato del tool di retrieval e
sostituisce la specie con una di sua invenzione (query su "nudibranch" -> il tool
restituisce "echinometra mathaei" -> il modello risponde comunque su "Aeolidia
papillosa"). Per eliminare questo rischio, il retrieval avviene in modo deterministico
in Python (retrieve_top_species) PRIMA di costruire la crew: la specie e' gia' fissata
quando il tassonomo entra in gioco, il suo compito e' solo descriverla bene.

L'LLM e' configurato per puntare a un server locale compatibile OpenAI (LM Studio di
default). Nessuna chiave API reale e' richiesta: il client OpenAI-compatibile ne
vuole comunque una stringa non vuota, per questo motivo se ne passa una fittizia.
"""

import os

from crewai import LLM, Agent, Crew, Process, Task

from .tools import EnvironmentalDataTool, retrieve_top_species

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "openai/gpt-oss-20b")

llm = LLM(
    model=LM_STUDIO_MODEL,
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    temperature=0.3,
)

environmental_tool = EnvironmentalDataTool()

taxonomist = Agent(
    role="Tassonomo marino",
    goal="Descrivere in modo chiaro e accurato la specie marina che ti viene indicata.",
    backstory=(
        "Sei un biologo marino esperto di tassonomia. Ricevi sempre una specie GIA' "
        "IDENTIFICATA da un sistema di retrieval a monte, insieme a un documento con "
        "tassonomia e descrizione visiva. Il tuo compito e' riformulare quel documento in "
        "una descrizione chiara — non sostituire mai la specie indicata con un'altra, "
        "anche se ti sembra di conoscerne una piu' specifica o piu' nota."
    ),
    llm=llm,
    verbose=True,
)

environmental_analyst = Agent(
    role="Analista ambientale",
    goal="Fornire dati quantitativi precisi sulla specie identificata.",
    backstory=(
        "Sei un data analyst specializzato in biodiversita' marina. Non inventi mai numeri: "
        "usi solo il tool di dati strutturati, con l'etichetta esatta che ti viene fornita "
        "nel task, e se un dato non e' disponibile lo dichiari esplicitamente."
    ),
    tools=[environmental_tool],
    llm=llm,
    verbose=True,
)

communicator = Agent(
    role="Comunicatore scientifico",
    goal="Sintetizzare le informazioni raccolte in una risposta chiara per l'utente finale.",
    backstory=(
        "Sei un divulgatore scientifico. Scrivi risposte comprensibili anche a un pubblico "
        "non specialista, ma non ometti mai numeri o incertezze riportate dai colleghi."
    ),
    llm=llm,
    verbose=True,
)


def build_crew(user_query: str, retrieved: dict) -> Crew:
    canonical_label = retrieved["canonical_label"]
    document = retrieved["document"]

    describe_task = Task(
        description=(
            f"Il sistema di retrieval ha identificato questa specie come la piu' pertinente "
            f'alla domanda "{user_query}": "{canonical_label}".\n\n'
            f"Documento associato (tassonomia, descrizione visiva, riepilogo ambientale):\n{document}\n\n"
            "Riformula queste informazioni in una descrizione chiara. NON sostituire la specie "
            f"'{canonical_label}' con un'altra, anche se ti risultasse piu' familiare o precisa."
        ),
        expected_output=f"Descrizione tassonomica e visiva di '{canonical_label}', basata solo sul documento fornito.",
        agent=taxonomist,
    )

    environmental_task = Task(
        description=(
            f"Recupera i dati ambientali strutturati (occorrenze, anomalie, copertura) per la "
            f"specie con etichetta canonica ESATTA: '{canonical_label}'. Usa questa stringa "
            "esatta come argomento del tool, senza modificarla."
        ),
        expected_output="Dati strutturati sulla specie: occorrenze OBIS, anni anomali, copertura ambientale.",
        agent=environmental_analyst,
    )

    synthesis_task = Task(
        description=(
            f'Domanda originale dell\'utente: "{user_query}"\n'
            "Componi una risposta finale in linguaggio naturale, chiara e accurata, che integri "
            "la descrizione tassonomica e i dati ambientali raccolti dai colleghi. Se un dato non "
            "era disponibile, dillo esplicitamente invece di ometterlo o inventarlo."
        ),
        expected_output="Risposta finale in linguaggio naturale per l'utente.",
        agent=communicator,
        context=[describe_task, environmental_task],
    )

    return Crew(
        agents=[taxonomist, environmental_analyst, communicator],
        tasks=[describe_task, environmental_task, synthesis_task],
        process=Process.sequential,
        verbose=True,
    )


async def ask(user_query: str) -> str:
    """Esegue l'intera pipeline: retrieval deterministico + crew per descrizione/analisi/sintesi."""
    retrieved = retrieve_top_species(user_query)
    if retrieved is None:
        return "Non ho trovato nessuna specie pertinente a questa domanda nella base di conoscenza."

    result = await build_crew(user_query, retrieved).kickoff_async()
    return str(result)
