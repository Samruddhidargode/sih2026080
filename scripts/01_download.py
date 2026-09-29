"""Download everything the pipeline needs. Resumable: every artefact is cached per file.

    python scripts/01_download.py                 # all years/leads in config
    python scripts/01_download.py --years 2025    # subset
    python scripts/01_download.py --workers 8
"""
import argparse, logging, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed  # processes: eccodes is not thread-safe
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PYTHONUTF8", "1")

from rainfall.config import load_config
from rainfall.data.gfs import build_gfs_sample, season_inits
from rainfall.data.imd import download_imd_rain
from rainfall.data.districts import load_districts

log = logging.getLogger("download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--leads", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-gfs", action="store_true")
    ap.add_argument("--skip-dem", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    years = a.years or sorted(set(cfg["years"]["train"] + cfg["years"]["calib"] + cfg["years"]["test"]))
    leads = a.leads or cfg["gfs"]["leads_days"]

    log.info("IMD rain %s", years)
    download_imd_rain(years, cfg["paths"]["raw"])

    log.info("districts")
    gdf = load_districts(cfg)
    log.info("%d districts", len(gdf))

    if not a.skip_dem:
        from rainfall.data.dem import load_terrain
        log.info("terrain")
        load_terrain(cfg)

    if a.skip_gfs:
        return
    inits = season_inits(years, cfg["season"]["months"], cfg["gfs"]["run_hour"])
    jobs = [(i, l) for i in inits for l in leads]
    log.info("GFS: %d (init, lead) samples with %d workers", len(jobs), a.workers)
    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build_gfs_sample, i, l, cfg): (i, l) for i, l in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            i, l = futs[f]
            try:
                f.result()
            except Exception as e:
                failed.append((str(i.date()), l, repr(e)[:120]))
                log.warning("FAILED %s L%d: %s", i.date(), l, repr(e)[:120])
            if n % 20 == 0:
                log.info("%d/%d done, %.0f s elapsed, %d failed", n, len(jobs), time.time() - t0, len(failed))
    out = Path(cfg["paths"]["interim"]) / "gfs_failed.csv"
    out.write_text("init,lead,error\n" + "\n".join(",".join(map(str, r)) for r in failed))
    log.info("GFS done: %d ok, %d failed (see %s)", len(jobs) - len(failed), len(failed), out)


if __name__ == "__main__":
    main()
