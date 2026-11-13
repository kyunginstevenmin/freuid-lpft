import csv
from pathlib import Path


def generate_csv(images_dir, path_to_dump):
    rows = []
    for f in sorted(images_dir.rglob("*.jpg")):
        is_attack = "True" if f.parent.name == "fakes" else "False"
        rows.append((f.stem, str(f.relative_to(images_dir.parent)), is_attack))

    with open(path_to_dump, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow(["id", "path", "is_attack"])
        writer.writerows(rows)


def main():
    # run from repo root; deterministic (sorted walk)
    data_dir = Path("sources/SIDT_clips_cropped")
    generate_csv(data_dir / "Images", path_to_dump=data_dir / "images.csv")


if __name__ == "__main__":
    main()
