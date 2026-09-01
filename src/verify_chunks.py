# src/verify_chunks.py
"""
Quick verification for structure-aware match chunks.

Checks:
  - total counts (parents vs children)
  - every child has a parent_id
  - parent_id actually exists
  - required metadata fields
  - sample matches look correct (lineups, goals, etc.)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "match_chunks.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    if not CHUNKS_PATH.exists():
        print(f"File not found: {CHUNKS_PATH}")
        return

    chunks = load_jsonl(CHUNKS_PATH)
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH.name}\n")

    # ------------------------------------------------------------------
    # 1. Basic counts
    # ------------------------------------------------------------------
    type_counts = Counter(c["metadata"].get("chunk_type") for c in chunks)
    parents = [c for c in chunks if c["metadata"].get("chunk_type") == "parent"]
    children = [c for c in chunks if c["metadata"].get("chunk_type") != "parent"]

    print("=== Chunk type counts ===")
    for t, n in sorted(type_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {t:15} {n:,}")
    print(f"\n  Parents : {len(parents):,}")
    print(f"  Children: {len(children):,}")

    # ------------------------------------------------------------------
    # 2. Parent–child integrity
    # ------------------------------------------------------------------
    parent_ids = {p["id"] for p in parents}
    missing_parent_id = 0
    broken_parent_link = 0

    for c in children:
        pid = c["metadata"].get("parent_id")
        if not pid:
            missing_parent_id += 1
        elif pid not in parent_ids:
            broken_parent_link += 1

    print("\n=== Parent–child integrity ===")
    print(f"  Children missing parent_id : {missing_parent_id}")
    print(f"  Children with broken link  : {broken_parent_link}")

    # ------------------------------------------------------------------
    # 3. Required metadata fields
    # ------------------------------------------------------------------
    required = ["match_id", "year", "team1", "team2", "chunk_type"]
    missing_fields: Counter[str] = Counter()

    for c in chunks:
        meta = c.get("metadata", {})
        for field in required:
            if field not in meta or meta[field] is None:
                missing_fields[field] += 1

    print("\n=== Required metadata ===")
    if missing_fields:
        for field, n in missing_fields.most_common():
            print(f"  Missing {field}: {n}")
    else:
        print("  All required fields present ✓")

    # ------------------------------------------------------------------
    # 4. Chunks per match (should be >= 2: parent + at least overview)
    # ------------------------------------------------------------------
    by_match: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        mid = c["metadata"].get("match_id", "unknown")
        by_match[mid].append(c["metadata"].get("chunk_type", "?"))

    sizes = [len(v) for v in by_match.values()]
    print("\n=== Chunks per match ===")
    print(f"  Matches          : {len(by_match):,}")
    print(f"  Min chunks/match : {min(sizes)}")
    print(f"  Max chunks/match : {max(sizes)}")
    print(f"  Avg chunks/match : {sum(sizes) / len(sizes):.1f}")

    # ------------------------------------------------------------------
    # 5. Sample matches (first + a modern final if present)
    # ------------------------------------------------------------------
    print("\n=== Sample: first match ===")
    first_match_id = parents[0]["metadata"]["match_id"] if parents else None
    if first_match_id:
        show_match(chunks, first_match_id)

    # Try to find a recent final
    final_parent = next(
        (p for p in reversed(parents) if p["metadata"].get("is_final")),
        None,
    )
    if final_parent and final_parent["metadata"]["match_id"] != first_match_id:
        print("\n=== Sample: recent final ===")
        show_match(chunks, final_parent["metadata"]["match_id"])

    # ------------------------------------------------------------------
    # 6. Quick quality checks on lineups
    # ------------------------------------------------------------------
    print("\n=== Lineup quality spot-check ===")
    lineup_chunks = [c for c in chunks if c["metadata"].get("chunk_type") == "lineup"]
    if lineup_chunks:
        sample = lineup_chunks[0]
        team = sample["metadata"].get("team")
        players = sample["metadata"].get("players", [])
        print(f"  Example team : {team}")
        print(f"  Players      : {len(players)}")
        print(f"  First 3      : {players[:3]}")

        # Check that the same player is not listed under both teams
        # for a single match (common previous bug)
        bad = 0
        for mid, types in by_match.items():
            if types.count("lineup") < 2:
                continue
            teams_players = {}
            for c in chunks:
                if c["metadata"].get("match_id") != mid:
                    continue
                if c["metadata"].get("chunk_type") != "lineup":
                    continue
                t = c["metadata"].get("team")
                teams_players[t] = set(c["metadata"].get("players", []))

            if len(teams_players) == 2:
                t1, t2 = list(teams_players.keys())
                overlap = teams_players[t1] & teams_players[t2]
                if overlap:
                    bad += 1

        print(f"  Matches with overlapping lineups: {bad}")
        if bad == 0:
            print("  Lineup assignment looks clean ✓")
    else:
        print("  No lineup chunks found")

    print("\nDone.")


def show_match(chunks: list[dict[str, Any]], match_id: str) -> None:
    related = [c for c in chunks if c["metadata"].get("match_id") == match_id]
    related.sort(key=lambda c: (
        0 if c["metadata"].get("chunk_type") == "parent" else 1,
        c["metadata"].get("chunk_type", ""),
        c["metadata"].get("chunk_index", 0),
    ))

    print(f"  match_id: {match_id}")
    for c in related:
        ctype = c["metadata"].get("chunk_type")
        preview = c["text"][:80].replace("\n", " ")
        print(f"    [{ctype:14}] {preview}...")


if __name__ == "__main__":
    main()
