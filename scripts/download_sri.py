import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.sri_scraper import DEFAULT_YEARS, download_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga CSV de vehiculos nuevos del SRI")
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--force", action="store_true", help="re-descargar aunque no haya cambios")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = download_all(args.years, args.out_dir, args.force)

    failed = [r for r in results if r["status"] == "failed"]
    for r in results:
        print(f"{r['year']}: {r['status']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
