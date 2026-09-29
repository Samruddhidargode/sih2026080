"""Tiny config loader so every script/module shares one YAML."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # make all paths absolute relative to the repo root so scripts work from any cwd
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((ROOT / v).resolve())
    cfg["dem"]["tiles_dir"] = str((ROOT / cfg["dem"]["tiles_dir"]).resolve())
    cfg["dem"]["out"] = str((ROOT / cfg["dem"]["out"]).resolve())
    cfg["districts"]["path"] = str((ROOT / cfg["districts"]["path"]).resolve())
    return cfg
