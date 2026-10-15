"""App FastAPI di Marine AI.

Avvio (dalla root del progetto, con l'ambiente virtuale attivato):

    uvicorn marine_ai.api.main:app --reload --app-dir src

Il modello e i dati ambientali vengono caricati una sola volta all'avvio
(evento "startup"), non ad ogni richiesta.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .inference import SpeciesClassifier
from .routers import assistant, classify, environmental


def _find_project_root(marker: str = ".gitignore") -> Path:
    path = Path(__file__).resolve()
    while path != path.parent:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Marker '{marker}' non trovato risalendo da {Path(__file__).resolve()}")


PROJECT_ROOT = _find_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = MODELS_DIR / "marine_classifier_best.pt"
    if checkpoint_path.exists():
        app.state.classifier = SpeciesClassifier(checkpoint_path)
        print(f"Classificatore caricato (val_acc={app.state.classifier.val_accuracy}).")
    else:
        app.state.classifier = None
        print(f"ATTENZIONE: checkpoint non trovato in {checkpoint_path} — /api/classify restituira' 503.")

    from .routers.environmental import _load_dataframes
    app.state.environmental_data = _load_dataframes(DOCS_DIR)
    print("Dati ambientali caricati.")

    yield  # l'app gira qui

    app.state.classifier = None


app = FastAPI(title="Marine AI", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

app.include_router(classify.router)
app.include_router(environmental.router)
app.include_router(assistant.router)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/health")
def health(request: Request):
    return {
        "classifier_loaded": request.app.state.classifier is not None,
        "n_species_environmental": len(request.app.state.environmental_data["coverage"]),
    }
