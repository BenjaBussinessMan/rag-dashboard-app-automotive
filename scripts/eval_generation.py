"""Corre preguntas reales por el pipeline RAG completo y resume cuántas respuestas pasan el guardrail."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.ollama import DEFAULT_MODEL, OllamaClient  # noqa: E402
from src.generation.rag import REFUSAL, RAGPipeline  # noqa: E402
from src.retrieval.service import build_retriever  # noqa: E402

QUESTIONS = ROOT / "tests" / "eval" / "generation_questions.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-gpu", type=int, default=None, help="0 = solo CPU")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    llm = OllamaClient(args.model, num_gpu=args.num_gpu)
    llm.check()
    pipeline = RAGPipeline(build_retriever(rerank=False), llm, k=5)

    rows = []
    for item in json.loads(args.questions.read_text(encoding="utf-8")):
        t0 = time.time()
        ans = pipeline.answer(item["q"])
        rows.append({"q": item["q"], "answer": ans.text, "grounded": ans.verdict.grounded,
                     "retried": ans.retried, "refused": ans.text.startswith(REFUSAL),
                     "problems": ans.verdict.problems, "sources": [h.doc_id for h in ans.hits],
                     "expected": item.get("expected"), "seconds": round(time.time() - t0, 1)})
        r = rows[-1]
        estado = "OK" if r["grounded"] and not r["refused"] else ("RECHAZADA" if r["refused"] else "SIN RESPALDO")
        print(f"\n[{estado}{' +reintento' if r['retried'] else ''}] {r['q']}  ({r['seconds']} s)\n{r['answer']}\n"
              f"fuentes: {r['sources'][:3]}")

    n = len(rows)
    print(f"\nResumen ({n} preguntas, modelo {args.model}): "
          f"verificadas {sum(r['grounded'] and not r['refused'] for r in rows)}, "
          f"rechazadas {sum(r['refused'] for r in rows)}, con reintento {sum(r['retried'] for r in rows)}")
    (ROOT / "data" / "processed" / "generation_eval.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
