"""Caricamento del classificatore addestrato (notebook 03) ed esecuzione dell'inferenza.

Il modello viene caricato una sola volta all'avvio dell'app (vedi main.py, evento
"startup"), non ad ogni richiesta: il caricamento dei pesi e il trasferimento su GPU
hanno un costo che non ha senso ripetere per ogni singola immagine caricata.
"""

from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SpeciesClassifier:
    """Wrapper leggero attorno al checkpoint EfficientNet-B0 del notebook 03."""

    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.idx_to_label: dict[int, str] = checkpoint["idx_to_label"]
        num_classes = len(self.idx_to_label)

        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.val_accuracy = checkpoint.get("val_acc")

    @torch.no_grad()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[dict]:
        """Restituisce le top_k predizioni come lista di {label, confidence}."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = _inference_transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.idx_to_label)))

        return [
            {"label": self.idx_to_label[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]
