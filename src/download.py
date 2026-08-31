import json
from pathlib import Path

import requests


REPO_API_URL = (
    "https://api.github.com/repos/openfootball/worldcup.json"
    "/git/trees/master?recursive=1"
)

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_files():
    """Get only worldcup.json and worldcup-full.json paths."""
    response = requests.get(REPO_API_URL)
    response.raise_for_status()

    tree = response.json()["tree"]

    files = [
        item["path"]
        for item in tree
        if item["type"] == "blob"
        and item["path"].split("/")[-1]
        in {"worldcup.json", "worldcup-full.json"}
    ]

    return files


def download_file(path):
    """Download one JSON file and save it under data/."""
    url = RAW_BASE_URL + path

    response = requests.get(url)
    response.raise_for_status()

    output_path = DATA_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            response.json(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Downloaded: {path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = get_files()

    print(f"Found {len(files)} files")

    for path in files:
        download_file(path)

    print("Done!")


if __name__ == "__main__":
    main()