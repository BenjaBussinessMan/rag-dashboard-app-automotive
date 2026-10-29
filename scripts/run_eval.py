"""Corre todas las evaluaciones en orden y genera docs/EVALUACION.md. Pensado para ejecutarse en segundo plano."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "processed" / "eval_logs"
PY = sys.executable

# El modelo de 7B no cabe en 4 GB de VRAM: corre solo en CPU y los embeddings tambien, para dejar la GPU libre.
MODELS = {"qwen2.5:3b-instruct": {"args": [], "env": {}},
          "qwen2.5:7b-instruct": {"args": ["--num-gpu", "0"], "env": {"RAG_EMBED_DEVICE": "cpu"}}}


def run(name: str, cmd: list[str], env_extra: dict) -> bool:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_extra}
    t0 = time.time()
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
        code = subprocess.call([PY, *cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{'ok' if code == 0 else 'ERROR ' + str(code)}] {name} ({time.time() - t0:.0f} s)", flush=True)
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--skip", nargs="*", default=[], choices=["retrieval", "sql", "generation"])
    args = parser.parse_args()

    ok = True
    if "retrieval" not in args.skip:
        ok &= run("retrieval", ["scripts/eval_retrieval.py", "--verbose"], {})
    for model in args.models:
        cfg, tag = MODELS[model], model.replace(":", "_")
        for suite in ("sql", "generation"):
            if suite not in args.skip:
                ok &= run(f"{suite}_{tag}", [f"scripts/eval_{suite}.py", "--model", model, *cfg["args"]], cfg["env"])
    ok &= run("report", ["scripts/make_report.py"], {})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
