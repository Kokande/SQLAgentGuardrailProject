"""
Бенчмарк защитных барьеров.

Запуск из корня проекта:
    python -m tests.benchmark
    python -m tests.benchmark --guardrail regex
    python -m tests.benchmark --guardrail ml
    python -m tests.benchmark --guardrail llm      # требует GigaChat API
    python -m tests.benchmark --guardrail hybrid   # требует GigaChat API
    python -m tests.benchmark --all
"""

import sys
import asyncio
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

# Добавляем src/ в путь поиска модулей
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guardrail import BaseGuardrail, RegexGuardrail, MLGuardrail, LLMGuardrail
from guardrail.core import GuardrailResponse
from tests.dataset import TEST_CASES, UNSAFE_COUNT, SAFE_COUNT


class HybridGuardrail(BaseGuardrail):
    """RegEx → ML → LLM, блокирует при первом срабатывании."""

    def __init__(self, *guardrails: BaseGuardrail):
        self._guardrails = guardrails

    def __repr__(self) -> str:
        return f"HybridGuardrail({', '.join(repr(g) for g in self._guardrails)})"

    async def preprocess(self, message: str) -> GuardrailResponse:
        for g in self._guardrails:
            resp = await g.preprocess(message)
            if resp.blocked:
                return resp
        return GuardrailResponse(blocked=False, commentary="HybridPassed;")

    async def postprocess(self, message: str) -> GuardrailResponse:
        for g in self._guardrails:
            resp = await g.postprocess(message)
            if resp.blocked:
                return resp
        return GuardrailResponse(blocked=False, commentary="HybridPassed;")


@dataclass
class CaseResult:
    message: str
    true_label: str
    predicted_label: str
    category: str
    description: str
    latency_ms: float
    commentary: str


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


async def run_guardrail(
    guardrail: BaseGuardrail, phase: str = "preprocess"
) -> tuple[list[CaseResult], Metrics]:
    results: list[CaseResult] = []
    metrics = Metrics()

    for message, true_label, category, description in TEST_CASES:
        t0 = time.perf_counter()
        if phase == "preprocess":
            response = await guardrail.preprocess(message)
        else:
            response = await guardrail.postprocess(message)
        latency = (time.perf_counter() - t0) * 1000

        predicted = "unsafe" if response.blocked else "safe"
        result = CaseResult(
            message=message,
            true_label=true_label,
            predicted_label=predicted,
            category=category,
            description=description,
            latency_ms=latency,
            commentary=response.commentary,
        )
        results.append(result)
        metrics.latencies.append(latency)

        if true_label == "unsafe" and predicted == "unsafe":
            metrics.tp += 1
        elif true_label == "safe" and predicted == "unsafe":
            metrics.fp += 1
        elif true_label == "safe" and predicted == "safe":
            metrics.tn += 1
        else:
            metrics.fn += 1

    return results, metrics


def print_confusion_matrix(name: str, m: Metrics) -> None:
    print(f"\n{'=' * 52}")
    print(f"  {name}")
    print(f"{'=' * 52}")
    print("  Матрица ошибок:")
    print(f"  {'':20s} {'Pred: unsafe':>14} {'Pred: safe':>12}")
    print(f"  {'True: unsafe':20s} {'TP = ' + str(m.tp):>14} {'FN = ' + str(m.fn):>12}")
    print(f"  {'True: safe':20s} {'FP = ' + str(m.fp):>14} {'TN = ' + str(m.tn):>12}")
    print()
    print(f"  Precision : {m.precision:.4f}")
    print(f"  Recall    : {m.recall:.4f}")
    print(f"  F1        : {m.f1:.4f}")
    print(f"  Accuracy  : {m.accuracy:.4f}")
    print(f"  Avg latency: {m.avg_latency_ms:.2f} мс")


def print_case_details(results: list[CaseResult]) -> None:
    print("\n  Детализация по случаям:")
    print(f"  {'#':>3}  {'Статус':6}  {'Категория':20}  {'Сообщение'}")
    print(f"  {'-' * 3}  {'-' * 6}  {'-' * 20}  {'-' * 40}")
    for i, r in enumerate(results, 1):
        status = "OK" if r.true_label == r.predicted_label else "MISS"
        emoji = "" if r.true_label == r.predicted_label else " !"
        print(f"  {i:>3}  {status:6}  {r.category:20}  {r.message[:45]}{emoji}")


def print_summary_table(metrics_map: dict[str, Metrics]) -> None:
    print(f"\n{'=' * 72}")
    print("  СВОДНАЯ ТАБЛИЦА")
    print(f"{'=' * 72}")
    header = f"  {'Метод':25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Задержка':>12}"
    print(header)
    print(f"  {'-' * 25} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")
    for name, m in metrics_map.items():
        print(
            f"  {name:25} {m.precision:>10.4f} {m.recall:>8.4f} {m.f1:>8.4f}"
            f" {m.accuracy:>8.4f} {m.avg_latency_ms:>9.2f} мс"
        )
    print()


async def _wait_for_ml_model(timeout: float = 60.0) -> None:
    """Block until the DistilBERT background loader finishes (or times out)."""
    import guardrail.ml_gr as ml_gr_module  # noqa: PLC0415

    deadline = asyncio.get_event_loop().time() + timeout
    while ml_gr_module._bert_available is None:
        if asyncio.get_event_loop().time() >= deadline:
            print(
                f"  [WARN] DistilBERT не загрузился за {timeout:.0f} с; используется keyword-scoring."
            )
            break
        await asyncio.sleep(0.5)
    backend = "DistilBERT" if ml_gr_module._bert_available else "keyword-scoring"
    print(f"  [ML] Backend: {backend}")


async def benchmark_one(name: str, guardrail: BaseGuardrail, verbose: bool) -> Metrics:
    if name in ("MLGuardrail", "Hybrid"):
        await _wait_for_ml_model()
    results, metrics = await run_guardrail(guardrail, phase="preprocess")
    print_confusion_matrix(name, metrics)
    if verbose:
        print_case_details(results)
    return metrics


async def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrail benchmark")
    parser.add_argument(
        "--guardrail",
        choices=["regex", "ml", "llm", "hybrid"],
        help="Конкретный метод для тестирования",
    )
    parser.add_argument("--all", action="store_true", help="Тестировать все методы")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Показать детализацию по случаям"
    )
    args = parser.parse_args()

    print(
        f"\nТестовая выборка: {len(TEST_CASES)} случаев "
        f"({UNSAFE_COUNT} unsafe / {SAFE_COUNT} safe)"
    )

    guardrails: dict[str, BaseGuardrail] = {}

    if args.all or args.guardrail == "regex" or not args.guardrail:
        guardrails["RegexGuardrail"] = RegexGuardrail()

    if args.all or args.guardrail == "ml":
        guardrails["MLGuardrail"] = MLGuardrail()

    llm_gr = None
    if args.all or args.guardrail in ("llm", "hybrid"):
        try:
            from dotenv import load_dotenv
            from agent.config import LLMConfig

            load_dotenv("src/configs/service.properties")
            load_dotenv("src/configs/secret.properties")
            LLMConfig.load()
            llm_gr = LLMGuardrail()
            if args.all or args.guardrail == "llm":
                guardrails["LLMGuardrail"] = llm_gr
        except Exception as e:
            print(f"\n[WARN] LLMGuardrail недоступен: {e}")

    if (args.all or args.guardrail == "hybrid") and llm_gr is not None:
        guardrails["Hybrid"] = HybridGuardrail(RegexGuardrail(), MLGuardrail(), llm_gr)

    metrics_map: dict[str, Metrics] = {}
    for name, gr in guardrails.items():
        if gr is None:
            continue
        metrics_map[name] = await benchmark_one(name, gr, verbose=args.verbose)

    if len(metrics_map) > 1:
        print_summary_table(metrics_map)


if __name__ == "__main__":
    asyncio.run(main())
