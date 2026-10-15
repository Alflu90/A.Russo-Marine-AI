"""Endpoint per i dati ambientali: elenco specie, trend annuali, anomalie.

Legge direttamente i CSV prodotti dai notebook 02 e 04 (docs/), gia' validati.
Nessun accesso a OBIS/WoRMS in tempo reale qui: sono dati precalcolati, coerenti
con quanto presentato nel deck e nella sinossi.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/environmental", tags=["dati ambientali"])


def _load_dataframes(docs_dir: Path) -> dict[str, pd.DataFrame]:
    coverage = pd.read_csv(docs_dir / "species_environmental_coverage.csv").drop_duplicates(
        subset="canonical_label"
    )
    trends_path = docs_dir / "species_yearly_trends.csv"
    anomalies_path = docs_dir / "species_anomalies.csv"
    occurrences_path = docs_dir / "obis_occurrence_sample.csv"
    trends = pd.read_csv(trends_path) if trends_path.exists() else pd.DataFrame()
    anomalies = pd.read_csv(anomalies_path) if anomalies_path.exists() else pd.DataFrame()
    occurrences = pd.read_csv(occurrences_path) if occurrences_path.exists() else pd.DataFrame()
    return {"coverage": coverage, "trends": trends, "anomalies": anomalies, "occurrences": occurrences}


@router.get("/species")
def list_species(request: Request):
    """Elenco delle specie con copertura ambientale affidabile (per popolare un menu a tendina)."""
    data = request.app.state.environmental_data
    ready = data["coverage"][data["coverage"]["env_module_ready"] == True]  # noqa: E712
    return {"species": sorted(ready["canonical_label"].unique().tolist())}


@router.get("/trend/{canonical_label}")
def get_trend(canonical_label: str, request: Request):
    """Serie storica delle osservazioni annuali per una specie, pronta per un grafico a linee."""
    trends = request.app.state.environmental_data["trends"]
    if trends.empty:
        raise HTTPException(status_code=404, detail="Dati di trend non disponibili.")

    species_trend = trends[trends["canonical_label"] == canonical_label].sort_values("date_year")
    if species_trend.empty:
        raise HTTPException(status_code=404, detail=f"Nessun trend disponibile per '{canonical_label}'.")

    return {
        "canonical_label": canonical_label,
        "years": species_trend["date_year"].tolist(),
        "n_occurrences": species_trend["n_occurrences"].tolist(),
        "anomaly_years": species_trend.loc[species_trend["is_anomaly"], "date_year"].tolist(),
    }


@router.get("/summary/{canonical_label}")
def get_species_summary(canonical_label: str, request: Request):
    """Riepilogo strutturato (tassonomia + occorrenze) — stessa logica del tool CrewAI/MCP."""
    coverage = request.app.state.environmental_data["coverage"].set_index("canonical_label")
    if canonical_label not in coverage.index:
        raise HTTPException(status_code=404, detail=f"Specie '{canonical_label}' non trovata.")

    row = coverage.loc[canonical_label]
    return {
        "canonical_label": canonical_label,
        "aphia_id": None if pd.isna(row.get("aphia_id")) else int(row["aphia_id"]),
        "obis_occurrence_count": None if pd.isna(row.get("obis_occurrence_count")) else int(row["obis_occurrence_count"]),
        "env_module_ready": bool(row.get("env_module_ready")),
        "taxonomy_reliable": bool(row.get("taxonomy_reliable", True)),
    }


@router.get("/occurrences/{canonical_label}")
def get_occurrences(canonical_label: str, request: Request, limit: int = 2000):
    """Punti geografici campionati (lat/lon) per una specie, usati per la mappa di distribuzione.

    Il campione (fino a `limit` punti) e' quello gia' raccolto dal notebook 04, non una
    nuova chiamata a OBIS: la mappa mostra la stessa base dati usata per trend e anomalie.
    """
    occurrences = request.app.state.environmental_data["occurrences"]
    if occurrences.empty:
        raise HTTPException(status_code=404, detail="Dati di occorrenza non disponibili.")

    subset = occurrences[occurrences["canonical_label"] == canonical_label].dropna(
        subset=["decimalLatitude", "decimalLongitude"]
    )
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"Nessuna occorrenza geolocalizzata per '{canonical_label}'.")

    if len(subset) > limit:
        subset = subset.sample(limit, random_state=42)

    points = subset[["decimalLatitude", "decimalLongitude"]].rename(
        columns={"decimalLatitude": "lat", "decimalLongitude": "lon"}
    )
    return {"canonical_label": canonical_label, "points": points.to_dict(orient="records")}
