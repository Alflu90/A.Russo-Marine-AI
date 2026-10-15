"""Endpoint per la classificazione di un'immagine caricata dall'utente."""

from fastapi import APIRouter, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api/classify", tags=["classificatore"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("")
async def classify_image(request: Request, file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Formato non supportato ({file.content_type}). Usa JPEG, PNG o WebP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Immagine troppo grande (limite 10 MB).")

    classifier = request.app.state.classifier
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificatore non disponibile: verifica che models/marine_classifier_best.pt esista.",
        )

    try:
        predictions = classifier.predict(image_bytes, top_k=3)
    except Exception as e:  # immagine corrotta o illeggibile, non un errore del server
        raise HTTPException(status_code=400, detail=f"Impossibile elaborare l'immagine: {e}")

    return {"predictions": predictions, "model_val_accuracy": classifier.val_accuracy}
