import io

import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.dataset import get_transforms
from src.model import build_model

DISCLAIMER = (
    "Research and educational demo only. This is not a medical device "
    "and does not provide a diagnosis."
)

DISPLAY_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

_transform = get_transforms(train=False)


def load_model(checkpoint_name: str = "best_finetuned.pt"):
    """Build the network without downloading weights, then load our trained ones."""
    model = build_model(freeze_backbone=False, pretrained=False)
    state = torch.load(config.CHECKPOINT_DIR / checkpoint_name, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_image(model, image_bytes: bytes) -> dict:
    """Take raw image bytes and return the prediction and class probabilities."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    batch = _transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)[0]

    top = int(probs.argmax())
    return {
        "prediction": config.CLASS_NAMES[top],
        "prediction_name": DISPLAY_NAMES[config.CLASS_NAMES[top]],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(config.CLASS_NAMES, probs)
        },
        "disclaimer": DISCLAIMER,
    }