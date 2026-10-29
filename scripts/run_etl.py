import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.etl.clean import run_etl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ETL: CSV crudos del SRI a Parquet")
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_etl(args.raw_dir, args.out_dir, args.years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
