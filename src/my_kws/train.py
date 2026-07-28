"""Train the DS-CNN wake-word detector on Google Speech Commands v2.

Run from the project root:
    uv run python -m my_kws.train
    uv run python -m my_kws.train --epochs 30 --batch-size 128 --lr 1e-3

Design notes
------------
Class imbalance: "yes" is ~4% of training samples, so plain BCE would let
the model score ~96% accuracy by always predicting "not keyword". We fix
this with BCEWithLogitsLoss(pos_weight=neg/pos), which scales the loss on
positive samples so both classes contribute roughly equally.

Metrics: accuracy is meaningless at 4% positives, so we track precision,
recall, and F1 on the positive class. For a wake-word system, precision
maps to false-alarm rate (device wakes when it shouldn't) and recall maps
to missed detections (you say the word, nothing happens).

Early stopping: we keep the checkpoint with the best validation F1 and
stop if it hasn't improved for `--patience` epochs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .cached_datasets import CachedKeywordSpottingDataset
from .kws_datasets import KeywordSpottingDataset
from .model import DSCNN, count_parameters


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def compute_pos_weight(root: str | Path, positive_words: tuple[str, ...]) -> tuple[float, int, int]:
    """Count positive/negative training files by walking the dataset dir.

    Reuses the official split files so the counts match what the Dataset
    assigns to the train split. Returns (pos_weight, n_pos, n_neg).
    """
    root = Path(root)
    held_out: set[str] = set()
    for name in ("validation_list.txt", "testing_list.txt"):
        held_out.update(
            line.strip().replace("\\", "/")
            for line in (root / name).read_text().splitlines()
            if line.strip()
        )

    n_pos = n_neg = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "_background_noise_":
            continue
        is_pos = d.name in positive_words
        for f in d.glob("*.wav"):
            if f"{d.name}/{f.name}" in held_out:
                continue
            if is_pos:
                n_pos += 1
            else:
                n_neg += 1
    return n_neg / max(n_pos, 1), n_pos, n_neg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    """Return loss / precision / recall / F1 over one full pass of `loader`."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    tp = fp = fn = tn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.float().view(-1, 1).to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        n_batches += 1

        pred = (torch.sigmoid(logits) >= threshold)
        truth = y.bool()
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "loss": total_loss / max(n_batches, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DS-CNN wake-word detector")
    parser.add_argument("--root", default="data/raw/speech_commands")
    parser.add_argument("--positive-words", nargs="+", default=["yes"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5,
                        help="early-stop after this many epochs without val F1 improvement")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; keep 0 on Windows unless tested")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cache", action="store_true",
                        help="train from precomputed log-mel cache "
                             "(faster; drops waveform-level augmentation)")
    parser.add_argument("--cache-root", default="data/cache/logmel")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    positive_words = tuple(args.positive_words)

    # ---- data -------------------------------------------------------------
    if args.use_cache:
        print(f"using precomputed feature cache: {args.cache_root}")
        train_ds = CachedKeywordSpottingDataset(
            cache_root=args.cache_root, data_root=args.root, split="train",
            positive_words=positive_words, training=True,
        )
        val_ds = CachedKeywordSpottingDataset(
            cache_root=args.cache_root, data_root=args.root, split="val",
            positive_words=positive_words, training=False,
        )
    else:
        train_ds = KeywordSpottingDataset(
            root=args.root, split="train", positive_words=positive_words,
            training=True, seed=args.seed,
        )
        val_ds = KeywordSpottingDataset(
            root=args.root, split="val", positive_words=positive_words,
            training=False, seed=args.seed,
        )
    print(f"train samples: {len(train_ds):,} | val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- class imbalance ----------------------------------------------------
    pos_weight, n_pos, n_neg = compute_pos_weight(args.root, positive_words)
    print(f"train positives: {n_pos:,} | negatives: {n_neg:,} | pos_weight: {pos_weight:.1f}")

    # ---- model / optimizer --------------------------------------------------
    model = DSCNN().to(device)
    print(f"model parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_cached" if args.use_cache else ""
    best_ckpt = out_dir / f"dscnn_best{suffix}.pt"
    history_path = out_dir / f"training_history{suffix}.json"

    # ---- loop -----------------------------------------------------------------
    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_metrics['loss']:.4f} | "
            f"P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
            f"F1 {val_metrics['f1']:.3f} | {elapsed:.0f}s"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        history_path.write_text(json.dumps(history, indent=2))

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "positive_words": list(positive_words),
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"  ↳ new best F1 {best_f1:.3f}, checkpoint saved to {best_ckpt}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping: no val F1 improvement in {args.patience} epochs")
                break

    print(f"\ndone. best val F1: {best_f1:.3f} | checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
