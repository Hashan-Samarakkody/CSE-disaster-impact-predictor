"""Read and write the cached outputs that pass data between pipeline stages."""

from __future__ import annotations

import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.settings import (ARTIFACT_DIR, MODEL_DIR, RESULT_DIR, TABLE_DIR,
                                 ensure_output_directories)

_LIBRARIES = ("numpy", "pandas", "sklearn", "xgboost", "torch", "statsmodels",
              "shap", "matplotlib", "wbgapi", "yfinance")


def _directory_for(suffix: str) -> Path:
    """Tables, fitted models and plain result payloads live in separate folders."""
    known = {".parquet": TABLE_DIR, ".csv": TABLE_DIR,
             ".pkl": MODEL_DIR, ".json": RESULT_DIR}
    if suffix not in known:
        raise ValueError(
            f"No artifact directory for extension {suffix!r}. The cache holds "
            f"{', '.join(sorted(known))}. For a directory such as the figure folder, "
            f"import it from src.config.settings instead.")
    return known[suffix]


def artifact_path(name: str, suffix: str) -> Path:
    """Full path of a cached artifact, including its category directory."""
    return _directory_for(suffix) / f"{name}{suffix}"


def artifact_file(filename: str) -> Path:
    """Full path of a cached artifact given its plain filename.

    Routes by extension so callers never hard code the tables, models or results
    subdirectory. A provenance sidecar sits beside the file it describes.
    """
    name = filename
    if name.endswith(".provenance.json"):
        for suffix in (".parquet", ".pkl", ".csv"):
            candidate = _directory_for(suffix) / name
            if candidate.exists():
                return candidate
        return _directory_for(".json") / name
    suffix = Path(name).suffix
    return _directory_for(suffix) / name


def _provenance_record() -> dict:
    """Library versions and a UTC timestamp, recorded beside every artifact."""
    versions = {}
    for library in _LIBRARIES:
        try:
            versions[library] = __import__(library).__version__
        except Exception:
            versions[library] = "not installed"
    return {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "versions": versions}


def _write_provenance(name: str, suffix: str, extra: dict) -> None:
    record = _provenance_record()
    record.update(extra)
    path = _directory_for(suffix) / f"{name}.provenance.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def save_frame(df: pd.DataFrame, name: str, note: str = "") -> Path:
    """Write a DataFrame to the table cache with a provenance sidecar."""
    ensure_output_directories()
    path = artifact_path(name, ".parquet")
    df.to_parquet(path, index=False)
    _write_provenance(name, ".parquet", {"artifact": name, "rows": int(len(df)),
                                         "columns": list(map(str, df.columns)),
                                         "note": note})
    print(f"cached {path.name} ({len(df)} rows by {len(df.columns)} columns)")
    return path


def load_frame(name: str) -> pd.DataFrame:
    """Read a cached DataFrame, with a clear error if its stage has not run."""
    path = artifact_path(name, ".parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' is missing. Run the stage that produces it first. "
            f"See the stage table in docs/architecture.md.")
    return pd.read_parquet(path)


def save_object(obj, name: str, note: str = "") -> Path:
    """Pickle a Python object such as the nested per fold results dictionaries."""
    ensure_output_directories()
    path = artifact_path(name, ".pkl")
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_provenance(name, ".pkl", {"artifact": name, "note": note,
                                     "type": type(obj).__name__})
    print(f"cached {path.name}")
    return path


def load_object(name: str):
    """Read a pickled artifact, with a clear error if its stage has not run."""
    path = artifact_path(name, ".pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' is missing. Run the stage that produces it first.")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_json(payload: dict, name: str) -> Path:
    """Write a plain JSON result payload to the result cache."""
    ensure_output_directories()
    path = artifact_path(name, ".json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"cached {path.name}")
    return path


def load_json(name: str) -> dict:
    """Read a cached JSON result payload."""
    path = artifact_path(name, ".json")
    if not path.exists():
        raise FileNotFoundError(f"Artifact '{name}' is missing. Run the stage that writes it.")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_status() -> pd.DataFrame:
    """One row per cached artifact, printed at the top of every notebook stage."""
    rows = []
    for meta_path in sorted(ARTIFACT_DIR.rglob("*.provenance.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({"artifact": meta.get("artifact"),
                     "written_utc": meta.get("written_utc"),
                     "rows": meta.get("rows", ""), "note": meta.get("note", "")})
    if not rows:
        return pd.DataFrame(columns=["artifact", "written_utc", "rows", "note"])
    return pd.DataFrame(rows)
