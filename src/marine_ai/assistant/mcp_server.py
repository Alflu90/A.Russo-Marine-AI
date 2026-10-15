"""Server MCP — BOZZA, non produzione.

Espone due tool, riusando la stessa logica di tools.py, per dimostrare il concetto:
un client MCP (es. Claude Desktop, o un altro agente) puo' interrogare direttamente
la base di conoscenza e i dati ambientali di Marine AI, senza passare dalla crew
CrewAI.

Volutamente minimale (richiesta esplicita del professore, data la complessita' del
protocollo): nessuna autenticazione, nessun hardening per produzione, un solo
transport (stdio). Sviluppi futuri: transport HTTP per uso remoto, autenticazione,
un terzo tool per l'inferenza del classificatore CV su un'immagine caricata.

Nota implementativa: `mcp dev` carica questo file come script standalone (non come
parte del package marine_ai), quindi un import relativo (`from .tools import ...`)
fallisce con "attempted relative import with no known parent package". Si aggiunge
percio' esplicitamente `src/` al sys.path, basandosi sulla posizione del file
(non sulla working directory, che puo' variare a seconda di come viene lanciato).
"""

import sys
from pathlib import Path


def _find_project_root(marker: str = ".gitignore") -> Path:
    path = Path(__file__).resolve()
    while path != path.parent:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Marker '{marker}' non trovato risalendo da {Path(__file__).resolve()}")


_PROJECT_ROOT = _find_project_root()
_SRC_DIR = str(_PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mcp.server.fastmcp import FastMCP

from marine_ai.assistant.tools import EnvironmentalDataTool, retrieve_top_species

mcp = FastMCP("Marine AI — bozza")
_environmental_tool = EnvironmentalDataTool()


@mcp.tool()
def cerca_specie(domanda: str) -> str:
    """Cerca la specie marina piu' pertinente a una domanda in linguaggio naturale.

    Usa prima una corrispondenza letterale contro le etichette conosciute, poi il
    recupero semantico come riserva. Restituisce tassonomia, descrizione visiva e
    un riepilogo ambientale generico per la specie trovata.
    """
    retrieved = retrieve_top_species(domanda)
    if retrieved is None:
        return "Nessuna specie pertinente trovata nella base di conoscenza."
    return f"Specie identificata: {retrieved['canonical_label']}\n\n{retrieved['document']}"


@mcp.tool()
def dati_ambientali(etichetta_canonica: str) -> str:
    """Restituisce dati ambientali strutturati ed esatti per una specie.

    Richiede l'etichetta canonica della specie (ottenibile da cerca_specie), ma
    tollera piccole variazioni tramite matching approssimato.
    """
    return _environmental_tool._run(etichetta_canonica)


if __name__ == "__main__":
    mcp.run()
