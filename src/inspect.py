import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_structure(data, path="root"):
    """Return all dictionary key paths in a JSON object."""
    structures = set()

    if isinstance(data, dict):
        for key, value in data.items():
            key_path = f"{path}.{key}"
            structures.add(key_path)
            structures.update(get_structure(value, key_path))

    elif isinstance(data, list):
        for item in data:
            structures.update(get_structure(item, f"{path}[]"))

    return structures


def inspect_files(files):
    structures = {}

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        structures[str(file_path)] = get_structure(data)

    return structures


def compare_files(files):
    structures = inspect_files(files)

    all_structures = defaultdict(list)

    for file_path, structure in structures.items():
        structure_key = frozenset(structure)
        all_structures[structure_key].append(file_path)

    print(f"Files checked: {len(files)}")
    print(f"Different structures found: {len(all_structures)}")

    for i, (structure, file_list) in enumerate(all_structures.items(), 1):
        print(f"\n{'=' * 70}")
        print(f"Structure {i}")
        print(f"Files: {len(file_list)}")

        for file_path in file_list:
            print(f"  {file_path}")

        print("\nKeys:")
        for key in sorted(structure):
            print(f"  {key}")


def main():
    # Check each filename type separately
    worldcup_files = sorted(DATA_DIR.rglob("worldcup.json"))
    full_files = sorted(DATA_DIR.rglob("worldcup-full.json"))

    print("\n" + "#" * 70)
    print("worldcup.json")
    print("#" * 70)

    compare_files(worldcup_files)

    print("\n" + "#" * 70)
    print("worldcup-full.json")
    print("#" * 70)

    compare_files(full_files)


if __name__ == "__main__":
    main()