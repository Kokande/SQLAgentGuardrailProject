#!/usr/bin/env python3
"""
Translates the Nemotron Content Safety Dataset from English to Russian using
Helsinki-NLP/opus-mt-en-ru — a local, free, uncensored MarianMT model.

Features
--------
- Persistent state file — skips already-translated items; safe to interrupt and resume.
- In-memory translation cache — identical texts (common in violated_categories) are
  translated only once per run, cutting compute significantly.
- Long-text chunking — splits texts that exceed the model's 512-token limit on sentence
  boundaries before translating, then joins the parts.
- Dual logging: INFO to stdout, DEBUG to a rotating file inside the output directory.
- Preserves original item order and all non-translatable fields verbatim.

Translated fields: prompt, response, violated_categories.
Skipped values:    null, empty string, "REDACTED".
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import MarianMTModel, MarianTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
ORIGIN_DIR = _SCRIPT_DIR / "dataset" / "origin"
OUTPUT_DIR = _SCRIPT_DIR / "dataset" / "translated"
STATE_FILE = OUTPUT_DIR / ".translation_state.json"
LOG_FILE = OUTPUT_DIR / "translation.log"

DATASET_FILES = [
    "train.json",
    "refusals_train.json",
    "validation.json",
    "refusals_validation.json",
    "test.json",
]

TRANSLATABLE_FIELDS = ("prompt", "response", "violated_categories")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
MAX_CHUNK_TOKENS = 490  # Marian hard limit is 512; leave headroom for special tokens
MAX_RETRIES = 2

# ──────────────────────────────────────────────────────────────────────────────
# Logging — console (INFO) is set up immediately; file handler added after mkdir
# ──────────────────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("dataset_translate")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
logger.addHandler(_ch)


def _add_file_logging() -> None:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Model initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _init_llm() -> tuple[MarianMTModel, MarianTokenizer]:
    logger.info("Loading %s …", MODEL_NAME)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info("Model ready on %s", device)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            state = json.load(fh)
        logger.debug("Loaded state from %s", STATE_FILE)
        return state
    logger.debug("No existing state file; starting fresh.")
    return {}


def _save_state(state: dict) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Translation — with cache, chunking, and retry
# ──────────────────────────────────────────────────────────────────────────────
_translation_cache: dict[str, str] = {}


def _split_into_chunks(
    tokenizer: MarianTokenizer, text: str, max_tokens: int
) -> list[str]:
    """Split *text* into chunks that each fit within *max_tokens* tokens.

    Splits on sentence boundaries (blank lines → paragraphs, then newlines, then
    ". " sentence ends) to keep semantic units together.
    """
    # Fast path: fits in one chunk
    if len(tokenizer.encode(text)) <= max_tokens:
        return [text]

    # Split into candidate sentences, preserving separators
    import re

    sentences = re.split(r"(?<=\.)\s+|(?<=\n)\n+|\n", text)
    sentences = [s for s in sentences if s.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(tokenizer.encode(sentence))
        if current_len + sent_len > max_tokens and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0
        # If a single sentence exceeds the limit, hard-truncate it
        if sent_len > max_tokens:
            ids = tokenizer.encode(sentence)[:max_tokens]
            sentence = tokenizer.decode(ids, skip_special_tokens=True)
            sent_len = max_tokens
        current_parts.append(sentence)
        current_len += sent_len

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _translate_text(
    llm: tuple[MarianMTModel, MarianTokenizer], text: str
) -> Optional[str]:
    """Translate *text* locally via MarianMT with retry. Returns None on failure."""
    if text in _translation_cache:
        logger.debug("Cache hit for text: %.60r", text)
        return _translation_cache[text]

    model, tokenizer = llm

    for attempt in range(MAX_RETRIES):
        try:
            chunks = _split_into_chunks(tokenizer, text, MAX_CHUNK_TOKENS)
            parts: list[str] = []
            for chunk in chunks:
                inputs = tokenizer(
                    [chunk],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs)
                parts.append(tokenizer.decode(out[0], skip_special_tokens=True))
            result = " ".join(parts)
            _translation_cache[text] = result
            return result
        except Exception as exc:
            delay = 2**attempt
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %ds…",
                attempt + 1,
                MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)

    logger.error("All %d attempts failed for text: %.80r", MAX_RETRIES, text)
    return None


def _translate_item(
    llm: tuple[MarianMTModel, MarianTokenizer], item: dict
) -> tuple[dict, int, int]:
    """Translate all translatable fields of *item*.

    Returns (translated_item, api_calls_made, api_calls_failed).
    """
    result = dict(item)
    calls_made = calls_failed = 0

    for field in TRANSLATABLE_FIELDS:
        value = item.get(field)
        if not value or value == "REDACTED":
            continue

        calls_made += 1
        translated = _translate_text(llm, value)

        if translated is not None:
            result[field] = translated
            logger.debug(
                "  %-22s %.70r  →  %.70r",
                field + ":",
                value,
                translated,
            )
        else:
            calls_failed += 1
            logger.warning(
                "  Field '%s' could not be translated for item %s — original kept.",
                field,
                item.get("id"),
            )

    return result, calls_made, calls_failed


# ──────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────────────────
def _process_file(
    llm: tuple[MarianMTModel, MarianTokenizer], filename: str, state: dict
) -> None:
    src = ORIGIN_DIR / filename
    dst = OUTPUT_DIR / filename

    if not src.exists():
        logger.warning("Source file not found, skipping: %s", src)
        return

    logger.info("━━━ Processing %s ━━━", filename)

    with src.open("r", encoding="utf-8") as fh:
        items: list[dict] = json.load(fh)
    total = len(items)

    # Load any partially-translated output that already exists
    if dst.exists():
        with dst.open("r", encoding="utf-8") as fh:
            existing: list[dict] = json.load(fh)
        translated_by_id: dict[str, dict] = {it["id"]: it for it in existing}
    else:
        translated_by_id = {}

    file_state = state.setdefault(filename, {"translated_ids": []})
    done_ids: set[str] = set(file_state["translated_ids"])

    already_done = len(done_ids)
    remaining = total - already_done
    logger.info(
        "  Items: total=%d  already-translated=%d  remaining=%d",
        total,
        already_done,
        remaining,
    )

    if remaining == 0:
        logger.info("  Nothing to do — file fully translated.")
        return

    total_calls = total_failed = 0

    for idx, item in enumerate(items):
        item_id = item["id"]
        if item_id in done_ids:
            continue

        logger.info("  [%d/%d] Translating item %s…", idx + 1, total, item_id[:16])

        translated_item, calls, failed = _translate_item(llm, item)
        total_calls += calls
        total_failed += failed

        translated_by_id[item_id] = translated_item
        done_ids.add(item_id)
        file_state["translated_ids"] = list(done_ids)

        # Write output in original order so the file is always consistent
        ordered = [
            translated_by_id[it["id"]] for it in items if it["id"] in translated_by_id
        ]
        with dst.open("w", encoding="utf-8") as fh:
            json.dump(ordered, fh, ensure_ascii=False, indent=4)

        _save_state(state)

        logger.info(
            "    ✓  calls=%d  failed=%d  remaining=%d",
            calls,
            failed,
            total - len(done_ids),
        )

    logger.info(
        "  ━━ %s done  total_calls=%d  total_failed=%d  ━━",
        filename,
        total_calls,
        total_failed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _add_file_logging()

    separator = "=" * 62
    logger.info(separator)
    logger.info("Dataset Translation Script")
    logger.info("  Source : %s", ORIGIN_DIR)
    logger.info("  Output : %s", OUTPUT_DIR)
    logger.info("  State  : %s", STATE_FILE)
    logger.info("  Log    : %s", LOG_FILE)
    logger.info(separator)

    llm = _init_llm()
    state = _load_state()

    grand_total_files = 0
    for filename in DATASET_FILES:
        if (ORIGIN_DIR / filename).exists():
            grand_total_files += 1
            _process_file(llm, filename, state)

    logger.info(separator)
    logger.info("All %d file(s) processed.", grand_total_files)
    logger.info(
        "Translation cache hits this run: %d unique texts cached.",
        len(_translation_cache),
    )
    logger.info(separator)


if __name__ == "__main__":
    main()
