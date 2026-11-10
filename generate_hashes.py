
import hashlib
import json
from pathlib import Path

# Folder yang berisi seluruh gambar asli CRC-HGD-v1
DATASET_DIR = Path("dataset")

# File keluaran daftar hash
OUTPUT_FILE = Path("dataset_hashes.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {DATASET_DIR.resolve()}"
        )

    image_files = [
        path for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise ValueError("Tidak ditemukan gambar di dalam folder dataset.")

    hash_mapping = {}

    for index, image_path in enumerate(image_files, start=1):
        image_hash = calculate_sha256(image_path)

        # Simpan hash dan lokasi relatif gambar.
        hash_mapping[image_hash] = str(image_path.relative_to(DATASET_DIR))

        print(f"[{index}/{len(image_files)}] {image_path.name}")

    data = {
        "dataset_name": "CRC-HGD-v1",
        "total_images_found": len(image_files),
        "total_unique_hashes": len(hash_mapping),
        "hashes": hash_mapping
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print("\n====================================")
    print("PEMBUATAN HASH SELESAI")
    print("====================================")
    print(f"Jumlah gambar ditemukan : {len(image_files)}")
    print(f"Jumlah hash unik        : {len(hash_mapping)}")
    print(f"File hasil              : {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
