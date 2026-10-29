import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.ollama import DEFAULT_MODEL, OllamaClient, OllamaUnavailable  # noqa: E402
from src.generation.rag import RAGPipeline  # noqa: E402
from src.retrieval.service import build_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Pregunta al asistente (RAG local con Ollama)")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-gpu", type=int, default=None, help="0 = solo CPU")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="activa el reranker BGE (más lento)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    llm = OllamaClient(args.model, num_gpu=args.num_gpu)
    try:
        llm.check()
    except OllamaUnavailable as exc:
        print(exc)
        return 2

    pipeline = RAGPipeline(build_retriever(rerank=args.rerank), llm, k=args.k, rerank=args.rerank)
    ans = pipeline.answer(args.question)
    print(ans.text)
    print("\nFuentes:")
    for i, h in enumerate(ans.hits, start=1):
        print(f"  [{i}] {h.doc_id}  ({h.metadata.get('fuente')})")
    estado = "verificada" if ans.verdict.grounded else "NO verificada: " + "; ".join(ans.verdict.problems)
    print(f"\nRespuesta {estado}" + (" (tras un reintento)" if ans.retried else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
