import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Arranca la API del asistente")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, app_dir=str(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
