import asyncio
import concurrent.futures
import logging
import threading
from pathlib import Path

from .core import BaseGuardrail, GuardrailResponse

logger = logging.getLogger("guardrail.ml")

# Path to the fine-tuned DistilBERT model produced by data_science/train_pipeline.py
_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "data_science" / "train_state" / "final_model"
)
# Probability threshold for the "unsafe" class (index 1)
_BERT_THRESHOLD = 0.5
# Wall-clock budget per inference call; exceeded → keyword fallback for that message
_BERT_INFERENCE_TIMEOUT = 5.0

# ── lazy-loaded BERT assets ──────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None  # set to torch.device when model loads
_bert_available: bool | None = (
    None  # None = still loading, False = unavailable, True = ready
)

# Single worker isolates BERT from the rest of the thread pool and caps CPU use
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="bert-infer"
)


def _try_load_bert() -> bool:
    global _model, _tokenizer, _device, _bert_available
    if _bert_available is not None:
        return _bert_available
    if not _MODEL_PATH.exists():
        logger.warning(
            "DistilBERT model not found at %s; falling back to keyword scoring.",
            _MODEL_PATH,
        )
        _bert_available = False
        return False
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _device.type == "cpu":
            torch.set_num_threads(1)

        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_PATH))
        _model = AutoModelForSequenceClassification.from_pretrained(
            str(_MODEL_PATH)
        ).to(_device)
        _model.eval()
        logger.info(
            "Loaded DistilBERT safety model from %s on %s", _MODEL_PATH, _device
        )
        _bert_available = True
        return True
    except Exception as exc:
        logger.warning(
            "Failed to load DistilBERT model (%s); using keyword scoring.", exc
        )
        _bert_available = False
        return False


# Warm the model in the background so the first user message does not pay load latency
threading.Thread(target=_try_load_bert, daemon=True, name="bert-loader").start()


def _bert_is_unsafe(message: str) -> bool:
    import torch

    raw = _tokenizer(
        message,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in raw.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    unsafe_prob = torch.softmax(logits, dim=-1)[0][1].item()
    return unsafe_prob >= _BERT_THRESHOLD


# ── keyword-scoring fallback ─────────────────────────────────────────────────
# Weighted bag-of-words heuristic used when the trained model is unavailable.
_TOXIC_KEYWORDS: dict[str, float] = {
    # Prompt-injection signals
    "ignore": 0.6,
    "forget": 0.6,
    "pretend": 0.5,
    "игнорируй": 0.7,
    "забудь": 0.7,
    "притворись": 0.6,
    "инструкции": 0.4,
    "instructions": 0.4,
    # Destructive SQL vocabulary
    "drop": 0.8,
    "delete": 0.7,
    "truncate": 0.9,
    "update": 0.5,
    "insert": 0.5,
    "alter": 0.7,
    "удали": 0.8,
    "удалить": 0.8,
    "очисти": 0.7,
    # Role-override signals
    "ты теперь": 0.7,
    "you are now": 0.7,
    "new persona": 0.6,
    "новый бот": 0.6,
    "другой бот": 0.6,
}
_KW_THRESHOLD = 1.0


def _keyword_is_unsafe(message: str) -> bool:
    lower = message.lower()
    score = sum(w for kw, w in _TOXIC_KEYWORDS.items() if kw in lower)
    return score >= _KW_THRESHOLD


async def _is_unsafe_async(message: str) -> bool:
    """Check safety without blocking the event loop.

    Uses BERT via a dedicated thread-pool executor when the model is ready;
    falls back to keyword scoring if the model is unavailable, still loading,
    or if inference exceeds _BERT_INFERENCE_TIMEOUT.
    """
    if _bert_available is True:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_executor, _bert_is_unsafe, message),
                timeout=_BERT_INFERENCE_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("BERT inference failed (%s); using keyword scoring.", exc)
    return _keyword_is_unsafe(message)


# ── guardrail class ───────────────────────────────────────────────────────────


class MLGuardrail(BaseGuardrail):
    def __repr__(self) -> str:
        backend = f"DistilBERT({_device})" if _bert_available else "keyword-scoring"
        return f"MLGuardrail({backend})"

    async def preprocess(self, message: str) -> GuardrailResponse:
        if await _is_unsafe_async(message):
            return GuardrailResponse(
                blocked=True,
                commentary="Сообщение помечено как потенциально опасное.",
            )
        return GuardrailResponse(blocked=False, commentary="MLPassed;")

    async def postprocess(self, message: str) -> GuardrailResponse:
        if await _is_unsafe_async(message):
            return GuardrailResponse(
                blocked=True,
                commentary="Ответ помечен как потенциально опасный.",
            )
        return GuardrailResponse(blocked=False, commentary="MLPassed;")
