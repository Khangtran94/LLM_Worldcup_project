import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_field_paths(data, path="root"):
    """
    Find all dictionary field paths in a JSON object.

    Example:
        root.matches[].team1
        root.matches[].score.ft
        root.matches[].goals1[].name
    """
    fields = set()

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}"
            fields.add(current_path)

            if isinstance(value, dict):
                fields.update(get_field_paths(value, current_path))

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        fields.update(
                            get_field_paths(item, f"{current_path}[]")
                        )

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                fields.update(get_field_paths(item, f"{path}[]"))

    return fields


def inspect_files(files):
    """
    Analyze which fields exist in each file.
    """
    field_files = defaultdict(set)

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        fields = get_field_paths(data)

        for field in fields:
            field_files[field].add(file_path)

    return field_files


def print_report(file_type, files, field_files):
    """
    Print common and optional fields.
    """
    total_files = len(files)

    print("\n" + "#" * 80)
    print(file_type)
    print("#" * 80)

    print(f"\nFiles checked: {total_files}")

    # Fields present in every file
    common_fields = [
        field
        for field, paths in field_files.items()
        if len(paths) == total_files
    ]

    # Fields present in some but not all files
    optional_fields = [
        field
        for field, paths in field_files.items()
        if len(paths) < total_files
    ]

    # Sort by path
    common_fields.sort()
    optional_fields.sort()

    print("\n" + "=" * 80)
    print("COMMON FIELDS")
    print("=" * 80)

    for field in common_fields:
        print(f"  {field}")

    print(f"\nTotal common fields: {len(common_fields)}")

    print("\n" + "=" * 80)
    print("OPTIONAL FIELDS")
    print("=" * 80)

    if optional_fields:
        print(
            f"{'Field':<55} {'Files':>8} {'Missing':>8}"
        )
        print("-" * 80)

        for field in optional_fields:
            count = len(field_files[field])
            missing = total_files - count

            print(
                f"{field:<55} "
                f"{count:>4}/{total_files:<3} "
                f"{missing:>8}"
            )

    print(f"\nTotal optional fields: {len(optional_fields)}")


def main():
    # Only these two exact filenames
    worldcup_files = sorted(
        DATA_DIR.rglob("worldcup.json")
    )

    full_files = sorted(
        DATA_DIR.rglob("worldcup-full.json")
    )

    # Analyze worldcup.json
    field_files = inspect_files(worldcup_files)

    print_report(
        "worldcup.json",
        worldcup_files,
        field_files,
    )

    # Analyze worldcup-full.json
    field_files = inspect_files(full_files)

    print_report(
        "worldcup-full.json",
        full_files,
        field_files,
    )


if __name__ == "__main__":
    main()