import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.dataset import get_dataloaders
from src.model import build_model


def predict(model, loader):
    """Run the model over a loader and return true labels and predictions."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            preds.extend(model(images).argmax(dim=1).tolist())
            labels.extend(y.tolist())
    return np.array(labels), np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint", default="best_model.pt")
    args = parser.parse_args()

    _, val_loader, test_loader = get_dataloaders()
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(freeze_backbone=False)
    state = torch.load(config.CHECKPOINT_DIR / args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    y_true, y_pred = predict(model, loader)
    labels = list(range(config.NUM_CLASSES))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=config.CLASS_NAMES,
        digits=3, zero_division=0,
    )
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{c}" for c in config.CLASS_NAMES],
        columns=[f"pred_{c}" for c in config.CLASS_NAMES],
    )

    print(f"Results on the {args.split} set ({len(y_true)} images)\n")
    print(report)
    print(cm.to_string())

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_tag = args.checkpoint.replace(".pt", "")
    out_path = config.OUTPUT_DIR / f"{args.split}_{ckpt_tag}_report.txt"
    out_path.write_text(report + "\n" + cm.to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()