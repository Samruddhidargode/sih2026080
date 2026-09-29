"""Assemble data/processed/dataset.zarr + norm_stats.json from cached GFS/IMD/terrain."""
import _common  # noqa: F401
from rainfall.config import load_config
from rainfall.data.build_dataset import build_and_save

if __name__ == "__main__":
    print(build_and_save(load_config()))
