"""Baixa o dataset Telco Customer Churn (IBM) para data/raw/."""

import argparse
import urllib.request
from pathlib import Path

from churn.config import DATASET_URL, RAW_DATA_PATH


def download_dataset(
    dest: Path = RAW_DATA_PATH, url: str = DATASET_URL, force: bool = False
) -> Path:
    if dest.exists() and not force:
        print(f"Dataset já existe em {dest} (use --force para baixar de novo).")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Salvo em {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="sobrescreve o arquivo existente")
    args = parser.parse_args()
    download_dataset(force=args.force)


if __name__ == "__main__":
    main()
