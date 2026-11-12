"""Read-only Company 2 adapter. Preserve identifier strings and source provenance."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_company2(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    tables, inventory = {}, {}
    for name, filename in manifest["entries"].items():
        path = directory / filename
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False,
                            na_values=[""], low_memory=False)
        expected = manifest["counts"][name]
        if len(frame) != expected:
            raise ValueError(f"{filename}: {len(frame)} rows, manifest expected {expected}")
        tables[name] = frame
        inventory[name] = {"filename": filename, "rows": len(frame), "columns": list(frame),
                           "sha256": sha256(path), "bytes": path.stat().st_size,
                           "missing": frame.isna().sum().astype(int).to_dict(),
                           "exact_duplicates": int(frame.duplicated().sum())}
    inventory["manifest_sha256"] = sha256(directory / "manifest.json")
    return tables, manifest, inventory
