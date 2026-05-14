#!/usr/bin/env python3
"""
Training pipeline for a binary safety classifier (safe / unsafe).

Uses the Nemotron Content Safety Dataset V2 stored under:
  data_science/dataset/origin/       — English source
  data_science/dataset/translated/   — Russian translation (may be partial)

Both folders are merged at the item level: if a translated counterpart exists
for a given sample it is used, otherwise the English text is kept.

Model default: distilbert-base-multilingual-cased (117M params, handles EN+RU).
Override with --model_name, e.g. xlm-roberta-base.

Checkpoints are saved to data_science/train_state/ and training automatically
resumes from the latest checkpoint on restart (disable with --no_resume).

Usage
-----
    python data_science/train_pipeline.py
    python data_science/train_pipeline.py --model_name xlm-roberta-base --num_epochs 5
    python data_science/train_pipeline.py --no_resume
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
ORIGIN_DIR = _SCRIPT_DIR / "dataset" / "origin"
TRANSLATED_DIR = _SCRIPT_DIR / "dataset" / "translated"
TRAIN_STATE_DIR = _SCRIPT_DIR / "train_state"

DATASET_FILES: dict[str, list[str]] = {
    "train": ["train.json", "refusals_train.json"],
    "validation": ["validation.json", "refusals_validation.json"],
    "test": ["test.json"],
}

LABEL2ID: dict[str, int] = {"safe": 0, "unsafe": 1}
ID2LABEL: dict[int, str] = {0: "safe", 1: "unsafe"}

DEFAULT_MODEL = "distilbert-base-multilingual-cased"

_LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("train_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_file: Path, level: str = "INFO") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_json_file(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.debug("Loaded %d items from %s", len(data), path.name)
    return data


def build_translated_lookup(translated_dir: Path, filename: str) -> dict[str, dict]:
    path = translated_dir / filename
    if not path.exists():
        logger.debug(
            "No translated counterpart for %s — will use origin text", filename
        )
        return {}
    items = load_json_file(path)
    return {item["id"]: item for item in items if "id" in item}


def merge_items(
    origin_items: list[dict], translated_lookup: dict[str, dict]
) -> list[dict]:
    merged = []
    translated_count = 0
    for item in origin_items:
        item_id = item.get("id")
        if item_id and item_id in translated_lookup:
            merged.append(translated_lookup[item_id])
            translated_count += 1
        else:
            merged.append(item)
    origin_only = len(origin_items) - translated_count
    logger.debug(
        "Merged %d items: %d translated, %d origin-only",
        len(merged),
        translated_count,
        origin_only,
    )
    return merged


def extract_text_and_label(item: dict) -> Optional[tuple[str, int]]:
    label_str = item.get("prompt_label")
    if label_str not in LABEL2ID:
        return None

    prompt = item.get("prompt")
    response = item.get("response")

    def _is_valid(text: Optional[str]) -> bool:
        return bool(text and text.strip() and text.strip() != "REDACTED")

    if _is_valid(prompt):
        text = prompt.strip()
    elif _is_valid(response):
        text = response.strip()
    else:
        return None

    return text, LABEL2ID[label_str]


def build_hf_dataset(
    filenames: list[str],
    origin_dir: Path,
    translated_dir: Path,
    split_name: str,
) -> "datasets.Dataset":
    import datasets as hf_datasets

    all_items: list[dict] = []
    for filename in filenames:
        origin_items = load_json_file(origin_dir / filename)
        if not origin_items:
            continue
        translated_lookup = build_translated_lookup(translated_dir, filename)
        merged = merge_items(origin_items, translated_lookup)
        all_items.extend(merged)

    texts: list[str] = []
    labels: list[int] = []
    skipped = 0
    for item in all_items:
        result = extract_text_and_label(item)
        if result is None:
            skipped += 1
            continue
        text, label = result
        texts.append(text)
        labels.append(label)

    safe_count = labels.count(0)
    unsafe_count = labels.count(1)
    logger.info(
        "Split %-10s — total=%d  safe=%d  unsafe=%d  skipped=%d",
        split_name,
        len(labels),
        safe_count,
        unsafe_count,
        skipped,
    )

    return hf_datasets.Dataset.from_dict({"text": texts, "label": labels})


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred: "transformers.EvalPrediction") -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_safe": float(
            f1_score(labels, preds, pos_label=0, average="binary", zero_division=0)
        ),
        "f1_unsafe": float(
            f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(labels, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, preds, average="macro", zero_division=0)
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint detection
# ──────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (
            p
            for p in output_dir.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    logger.info("Proccess start")
    import torch

    # transformers 5.x uses `from __future__ import annotations` in moe.py, turning all
    # type hints into strings ('torch.Tensor' instead of torch.Tensor).  torch 2.4.x
    # infer_schema only accepts actual type objects, so registration crashes at import time.
    # Fix: resolve string annotations via typing.get_type_hints() before registering.
    if hasattr(torch.library, "custom_op"):
        import functools as _functools
        import typing as _typing

        _orig_custom_op = torch.library.custom_op

        def _lenient_custom_op(
            qualname, fn=None, /, *, mutates_args=(), device_types=None, schema=None
        ):
            def _wrap(f):
                # Resolve forward-reference annotations to real types using the
                # function's own module globals (where torch is already imported).
                try:
                    hints = _typing.get_type_hints(f)

                    @_functools.wraps(f)
                    def _resolved(*args, **kwargs):
                        return f(*args, **kwargs)

                    _resolved.__annotations__ = hints
                    f = _resolved
                except Exception:
                    pass
                try:
                    return _orig_custom_op(
                        qualname,
                        f,
                        mutates_args=mutates_args,
                        device_types=device_types,
                        schema=schema,
                    )
                except (ValueError, TypeError):
                    return f

            return _wrap(fn) if fn is not None else _wrap

        torch.library.custom_op = _lenient_custom_op

    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    logger.info("Successful imports")

    output_dir = Path(args.output_dir)

    set_seed(args.seed)

    # ── Hardware info ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 62)
    logger.info("Safety Classifier Training Pipeline")
    logger.info("=" * 62)
    logger.info("Device          : %s", device)
    if device == "cuda":
        logger.info("CUDA device     : %s", torch.cuda.get_device_name(0))
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("VRAM            : %.1f GB", vram_gb)
        logger.info("CUDA version    : %s", torch.version.cuda)
    logger.info("PyTorch         : %s", torch.__version__)
    logger.info("Transformers    : %s", transformers.__version__)
    logger.info("Model           : %s", args.model_name)
    logger.info("Max seq length  : %d", args.max_length)
    logger.info("Epochs          : %d", args.num_epochs)
    logger.info(
        "Batch size      : %d (grad_accum=%d → effective=%d)",
        args.batch_size,
        args.grad_accum,
        args.batch_size * args.grad_accum,
    )
    logger.info("Learning rate   : %g", args.learning_rate)
    logger.info("Checkpoint dir  : %s", output_dir)
    logger.info("=" * 62)

    origin_dir = Path(args.origin_dir)
    translated_dir = Path(args.translated_dir)

    # ── Build datasets ─────────────────────────────────────────────────────────
    logger.info("Loading datasets…")
    train_dataset = build_hf_dataset(
        DATASET_FILES["train"], origin_dir, translated_dir, "train"
    )
    val_dataset = build_hf_dataset(
        DATASET_FILES["validation"], origin_dir, translated_dir, "validation"
    )
    test_dataset = build_hf_dataset(
        DATASET_FILES["test"], origin_dir, translated_dir, "test"
    )

    # ── Tokenise ───────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    logger.info("Tokenizing datasets (this may take a minute)…")
    tokenize_kwargs = dict(batched=True, remove_columns=["text"], desc="Tokenizing")
    train_dataset = train_dataset.map(tokenize_fn, **tokenize_kwargs)
    val_dataset = val_dataset.map(tokenize_fn, **tokenize_kwargs)
    test_dataset = test_dataset.map(tokenize_fn, **tokenize_kwargs)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded — %.1fM parameters", param_count)

    # ── FP16 decision ──────────────────────────────────────────────────────────
    if args.fp16 is None:
        use_fp16 = device == "cuda"
    else:
        use_fp16 = args.fp16
    logger.info("Mixed precision (fp16): %s", use_fp16)

    # ── Checkpoint resume ──────────────────────────────────────────────────────
    checkpoint: Optional[str] = None
    if not args.no_resume:
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            logger.info("Resuming from checkpoint: %s", checkpoint)
        else:
            logger.info("No existing checkpoint found — starting fresh")
    else:
        logger.info("--no_resume set — starting fresh (ignoring any checkpoints)")

    # ── TrainingArguments ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=use_fp16,
        bf16=False,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("Starting training…")
    trainer.train(resume_from_checkpoint=checkpoint)
    logger.info("Training complete.")

    # ── Save final model ───────────────────────────────────────────────────────
    final_model_dir = output_dir / "final_model"
    logger.info("Saving final model to %s", final_model_dir)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    logger.info("Model and tokenizer saved.")

    # ── Evaluate on test set ───────────────────────────────────────────────────
    logger.info("Evaluating on test set…")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    logger.info("=" * 62)
    logger.info("Test results:")
    for key, value in test_results.items():
        logger.info("  %-30s %.4f", key, value)
    logger.info("=" * 62)
    logger.info("Pipeline finished. Final model: %s", final_model_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary safety classifier on the Nemotron Content Safety dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID (e.g. xlm-roberta-base)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(TRAIN_STATE_DIR),
        help="Directory for checkpoints and final model",
    )
    parser.add_argument(
        "--log_file",
        default=str(TRAIN_STATE_DIR / "train.log"),
        help="Path to the training log file",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Per-device training batch size"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Initial learning rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Tokenizer max sequence length"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--origin_dir",
        default=str(ORIGIN_DIR),
        help="Path to the origin (English) dataset directory",
    )
    parser.add_argument(
        "--translated_dir",
        default=str(TRANSLATED_DIR),
        help="Path to the translated (Russian) dataset directory",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing checkpoints and start fresh",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Console logging level",
    )

    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="Force enable mixed-precision FP16 training",
    )
    fp16_group.add_argument(
        "--no_fp16",
        dest="fp16",
        action="store_false",
        help="Force disable mixed-precision FP16 training",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(Path(args.log_file), args.log_level)
    try:
        train(args)
    except Exception:
        logger.exception("Fatal error during training")
        sys.exit(1)


if __name__ == "__main__":
    main()
