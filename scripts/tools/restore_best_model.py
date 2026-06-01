"""
Rebuild best_model/model.safetensors from repository chunks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "best_model"
OUTPUT_PATH = MODEL_DIR / "model.safetensors"
PART_GLOB = "model.safetensors.part-*"
EXPECTED_SIZE = 710_771_556
EXPECTED_SHA256 = "10bef7fd4da6a257e4b31ba884474dbc02fb967b5ffbb70befc02ae4884a7c9d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parts = sorted(MODEL_DIR.glob(PART_GLOB))
    if not parts:
        raise SystemExit(f"No checkpoint chunks found under {MODEL_DIR}")

    with OUTPUT_PATH.open("wb") as output:
        for part in parts:
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    output.write(chunk)

    actual_size = OUTPUT_PATH.stat().st_size
    actual_sha256 = sha256_file(OUTPUT_PATH)
    if actual_size != EXPECTED_SIZE or actual_sha256 != EXPECTED_SHA256:
        OUTPUT_PATH.unlink(missing_ok=True)
        raise SystemExit(
            "Rebuilt checkpoint failed integrity check: "
            f"size={actual_size}, sha256={actual_sha256}"
        )

    print(f"Restored {OUTPUT_PATH}")
    print(f"sha256={actual_sha256}")


if __name__ == "__main__":
    main()
