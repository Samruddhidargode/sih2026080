import os, sys, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
