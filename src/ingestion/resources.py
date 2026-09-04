# src/ingestion/resources.py
"""
dlt resources/transformers wrapping the existing download -> transform ->
chunk pipeline. Embedding + Postgres load stays OUT of dlt on purpose —
see embed_and_load.py, run separately after this pipeline completes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import dlt
import requests

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transform import transform_match, PROJECT_ROOT
from chunk_matches import build_chunks

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPO_API_URL = (
    "https://api.github.com/repos/openfootball/worldcup.json"
    "/git/trees/master?recursive=1"
)
RAW_BASE_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/"


def _list_full_json_paths() -> list[str]:
    """Same tree listing as the original download.py, filtered to worldcup-full.json."""
    response = requests.get(REPO_API_URL)
    response.raise_for_status()
    tree = response.json()["tree"]
    return [
        item["path"]
        for item in tree
        if item["type"] == "blob" and item["path"].endswith("worldcup-full.json")
    ]


@dlt.resource(name="raw_matches", write_disposition="merge", primary_key="year")
def raw_matches() -> Iterator[dict[str, Any]]:
    """
    One record per year's worldcup-full.json.

    Years already present under data/raw/<year>/worldcup-full.json are read
    from disk and NEVER re-downloaded. Only a genuinely new year (e.g. 2030
    after the next World Cup) triggers a network call. This is a hard rule:
    existing downloaded data is never touched, refreshed, or re-fetched.
    """
    for path in _list_full_json_paths():
        year_str = path.split("/", 1)[0]
        output_path = RAW_DIR / path

        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            response = requests.get(RAW_BASE_URL + path)
            response.raise_for_status()
            data = response.json()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        yield {"year": int(year_str), "source_file": path, "data": data}


@dlt.transformer(
    data_from=raw_matches,
    name="transformed_matches",
    write_disposition="merge",
    primary_key="id",
)
def transformed_matches(raw_year_blob: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """One RAG-document dict per match, reusing transform.py's builder unchanged."""
    year = raw_year_blob["year"]
    source_file = RAW_DIR / raw_year_blob["source_file"]
    matches = raw_year_blob["data"].get("matches", [])

    for match_index, match in enumerate(matches, start=1):
        if not isinstance(match, dict):
            continue
        yield transform_match(year, match, source_file, match_index)


@dlt.transformer(
    data_from=transformed_matches,
    name="match_chunks",
    write_disposition="merge",
    primary_key="id",
)
def match_chunks(transformed_record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Structure-aware parent/child chunks, reusing chunk_matches.py unchanged."""
    yield from build_chunks(transformed_record)