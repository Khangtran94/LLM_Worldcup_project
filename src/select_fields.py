import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

MIN_FILE_RATIO = 10


def get_field_paths(data, path="root"):
    """Return all field paths found in a JSON object."""
    fields = set()

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}"
            fields.add(current_path)

            if isinstance(value, dict):
                fields.update(
                    get_field_paths(value, current_path)
                )

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        fields.update(
                            get_field_paths(
                                item,
                                f"{current_path}[]"
                            )
                        )

    return fields


def analyze_files(files):
    """Count how many files contain each field."""
    field_files = defaultdict(set)

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        fields = get_field_paths(data)

        for field in fields:
            field_files[field].add(file_path)

    return field_files


def print_selected_fields(file_type, files, field_files):
    total_files = len(files)

    selected = []
    ignored = []

    for field, paths in field_files.items():
        count = len(paths)

        if count >= MIN_FILE_RATIO:
            selected.append((field, count))
        else:
            ignored.append((field, count))

    selected.sort()
    ignored.sort()

    print("\n" + "#" * 80)
    print(file_type)
    print("#" * 80)

    print(f"\nFiles checked: {total_files}")
    print(f"Selection threshold: >= {MIN_FILE_RATIO}/{total_files}")

    print("\n" + "=" * 80)
    print("SELECTED FIELDS")
    print("=" * 80)

    for field, count in selected:
        print(f"  {field:<55} {count}/{total_files}")

    print(f"\nSelected fields: {len(selected)}")

    print("\n" + "=" * 80)
    print("IGNORED FIELDS")
    print("=" * 80)

    for field, count in ignored:
        print(f"  {field:<55} {count}/{total_files}")

    print(f"\nIgnored fields: {len(ignored)}")


def main():
    worldcup_files = sorted(
        DATA_DIR.rglob("worldcup.json")
    )

    full_files = sorted(
        DATA_DIR.rglob("worldcup-full.json")
    )

    worldcup_fields = analyze_files(worldcup_files)

    print_selected_fields(
        "worldcup.json",
        worldcup_files,
        worldcup_fields,
    )

    full_fields = analyze_files(full_files)

    print_selected_fields(
        "worldcup-full.json",
        full_files,
        full_fields,
    )


if __name__ == "__main__":
    main()