import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src import config
from src.dataset import get_dataloaders
from src.model import build_model, count_trainable_params, unfreeze_last_block


def compute_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    """Rarer classes get larger weights, so the loss doesn't ignore them."""
    counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
    weights = len(train_labels) / (config.NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def set_train_mode(model: nn.Module) -> None:
    """Train mode, but keep frozen BatchNorm layers fixed."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()


def run_epoch(model, loader, criterion, optimizer=None, max_batches=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    if training:
        set_train_mode(model)
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())

    n = len(all_labels)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per epoch, for quick tests")
    parser.add_argument("--finetune", action="store_true",
                        help="Start from best_model.pt and unfreeze the last block")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()
    model = build_model(freeze_backbone=True)

    if args.finetune:
        state = torch.load(config.CHECKPOINT_DIR / "best_model.pt", map_location="cpu")
        model.load_state_dict(state)
        unfreeze_last_block(model)
        lr = args.lr or 1e-4
        ckpt_name, history_name = "best_finetuned.pt", "history_finetune.json"
    else:
        lr = args.lr or config.LEARNING_RATE
        ckpt_name, history_name = "best_model.pt", "history.json"

    print(f"Trainable parameters: {count_trainable_params(model)} (lr={lr})")

    train_labels = train_loader.dataset.df["label"].astype(int).to_numpy()
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels))
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, args.max_batches
        )
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, None, args.max_batches
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
        })
        print(
            f"Epoch {epoch}/{args.epochs} ({time.time() - start:.0f}s) | "
            f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.3f} acc {va_acc:.3f} macro-F1 {va_f1:.3f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / ckpt_name)
            print("  saved new best model")

    with open(config.OUTPUT_DIR / history_name, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()