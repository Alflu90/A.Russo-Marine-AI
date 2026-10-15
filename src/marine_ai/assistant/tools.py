"""Tool usati dagli agenti CrewAI del modulo assistente.

Due strumenti con scopi deliberatamente diversi:
- SpeciesKnowledgeTool: retrieval semantico (RAG) sulla base di conoscenza costruita nel
  notebook 05 — utile per domande descrittive ("che aspetto ha", "che animale e'").
- EnvironmentalDataTool: lookup strutturato sui CSV del notebook 04 — utile per domande che
  richiedono un dato esatto (occorrenze, anni anomali), dove la similarita' testuale del RAG
  non e' affidabile (vedi i test di recupero del notebook 05 sul tema "avvistamenti insoliti").

Nota sul fuzzy matching in EnvironmentalDataTool: l'LLM tende a "migliorare" il nome canonico
con la propria conoscenza pregressa (es. riporta "Chelonia mydas" invece di "green sea turtle",
o inventa una specie precisa come "Aeolidia papillosa" al posto dell'etichetta ombrello
"nudibranch"). Il lookup esatto fallirebbe sempre in questi casi; per questo, se il nome esatto
non e' trovato, si tenta un matching approssimato contro le etichette canoniche note.
"""

import difflib
import re
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def find_project_root(marker: str = ".gitignore") -> Path:
    path = Path.cwd()
    while path != path.parent:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Marker '{marker}' non trovato risalendo da {Path.cwd()}")


PROJECT_ROOT = find_project_root()
DOCS_DIR = PROJECT_ROOT / "docs"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # deve coincidere col notebook 05

_chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8001)
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
_collection = _chroma_client.get_or_create_collection(
    name="marine_species_knowledge",
    embedding_function=_embedding_fn,
)

_coverage_df = (
    pd.read_csv(DOCS_DIR / "species_environmental_coverage.csv")
    .drop_duplicates(subset="canonical_label")
    .set_index("canonical_label")
)
_anomalies_path = DOCS_DIR / "species_anomalies.csv"
_anomalies_df = pd.read_csv(_anomalies_path) if _anomalies_path.exists() else pd.DataFrame()
_known_labels = list(_coverage_df.index)


def _resolve_canonical_label(raw_label: str):
    """Trova l'etichetta canonica piu' vicina a raw_label tra quelle note.

    Prova, in ordine: match esatto (case-insensitive) -> il nome noto e' contenuto nella
    stringa data (o viceversa) -> similarita' testuale approssimata (difflib). Restituisce
    None se nessun candidato supera la soglia minima di somiglianza.
    """
    if raw_label in _coverage_df.index:
        return raw_label

    lowered = raw_label.strip().lower()
    for label in _known_labels:
        if label.lower() == lowered:
            return label

    for label in _known_labels:
        if label.lower() in lowered or lowered in label.lower():
            return label

    close = difflib.get_close_matches(lowered, [l.lower() for l in _known_labels], n=1, cutoff=0.6)
    if close:
        for label in _known_labels:
            if label.lower() == close[0]:
                return label
    return None


class SpeciesKnowledgeInput(BaseModel):
    query: str = Field(..., description="Domanda o descrizione in linguaggio naturale su una specie marina")


def retrieve_top_species(query: str, n_results: int = 1):
    """Retrieval usato per fissare deterministicamente quale specie e' oggetto della
    domanda, PRIMA di generare qualunque testo con l'LLM.

    Fa prima un controllo di corrispondenza letterale (parola intera, case-insensitive)
    contro le etichette canoniche note: e' stato osservato empiricamente che il recupero
    semantico puro recupera male nomi di specie brevi dentro domande lunghe (es.
    "nudibranch" finiva 4a su 10 per la query "Ci sono state anomalie nelle osservazioni
    del nudibranch negli ultimi anni?", con distanze quasi indistinguibili dai risultati
    sbagliati). Le etichette piu' lunghe sono controllate per prime, cosi' un match piu'
    specifico (es. "green sea turtle") vince su uno piu' generico (es. "sea turtle") se
    entrambi compaiono nel testo. Solo se nessuna etichetta nota compare letteralmente
    nella domanda si ricorre alla ricerca semantica per similarita'.
    """
    lowered_query = query.lower()
    for label in sorted(_known_labels, key=len, reverse=True):
        pattern = r"\b" + re.escape(label.lower()) + r"\b"
        if re.search(pattern, lowered_query):
            fetched = _collection.get(ids=[label])
            document = fetched["documents"][0] if fetched["documents"] else label
            return {"canonical_label": label, "document": document, "distance": 0.0, "match_type": "literal"}

    results = _collection.query(query_texts=[query], n_results=n_results)
    if not results["ids"][0]:
        return None
    return {
        "canonical_label": results["ids"][0][0],
        "document": results["documents"][0][0],
        "distance": results["distances"][0][0],
        "match_type": "semantic",
    }


class SpeciesKnowledgeTool(BaseTool):
    name: str = "Ricerca nella base di conoscenza delle specie"
    description: str = (
        "Cerca, tra le specie marine indicizzate, quelle piu' pertinenti a una domanda o descrizione "
        "in linguaggio naturale. Restituisce tassonomia, descrizione visiva e un riepilogo ambientale "
        "generico per ciascuna specie trovata. Usalo per identificare QUALE specie e' rilevante — "
        "non e' affidabile per domande che richiedono un dato numerico esatto (per quello usa l'altro tool)."
    )
    args_schema: type[BaseModel] = SpeciesKnowledgeInput

    def _run(self, query: str) -> str:
        results = _collection.query(query_texts=[query], n_results=3)
        if not results["ids"][0]:
            return "Nessuna specie pertinente trovata nella base di conoscenza."
        blocks = [f"### {label}\n{doc}" for label, doc in zip(results["ids"][0], results["documents"][0])]
        return "\n\n".join(blocks)


class EnvironmentalDataInput(BaseModel):
    canonical_label: str = Field(..., description="Nome canonico esatto della specie, es. 'green sea turtle'")


class EnvironmentalDataTool(BaseTool):
    name: str = "Dati ambientali strutturati per specie"
    description: str = (
        "Restituisce dati esatti (non recuperati per similarita' testuale) su una specie: numero di "
        "occorrenze OBIS dichiarate, copertura tassonomica, anni con osservazioni anomale. Accetta il "
        "nome canonico della specie, ma tollera piccole variazioni (nome scientifico, maiuscole/minuscole) "
        "tramite un matching approssimato — se non trova nulla di simile, elenca le etichette disponibili."
    )
    args_schema: type[BaseModel] = EnvironmentalDataInput

    def _run(self, canonical_label: str) -> str:
        resolved = _resolve_canonical_label(canonical_label)
        if resolved is None:
            sample = ", ".join(_known_labels[:15])
            return (
                f"Specie '{canonical_label}' non trovata nei dati strutturati, nemmeno con "
                f"corrispondenza approssimata. Alcune etichette disponibili: {sample}, ..."
            )

        row = _coverage_df.loc[resolved]
        lines = [
            f"Specie (etichetta canonica usata): {resolved}",
            f"AphiaID (WoRMS): {row.get('aphia_id')}",
            f"Occorrenze OBIS totali (dichiarate): {row.get('obis_occurrence_count')}",
            f"Copertura ambientale affidabile: {'si' if row.get('env_module_ready') else 'no'}",
        ]

        if not _anomalies_df.empty:
            species_anomalies = _anomalies_df[_anomalies_df["canonical_label"] == resolved]
            if not species_anomalies.empty:
                years = sorted(species_anomalies["date_year"].tolist())
                lines.append(f"Anni con osservazioni anomale nel campione OBIS: {', '.join(map(str, years))}")
            else:
                lines.append("Nessuna anomalia rilevata nel campione analizzato per questa specie.")
        else:
            lines.append("Dati di anomalia non disponibili (notebook 04 non ancora eseguito?).")

        return "\n".join(lines)
