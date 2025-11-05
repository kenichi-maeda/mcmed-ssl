
import os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

# ================= Config =================
BASE = Path("physionet.org/files/mc-med/1.0.1/data")
WAVE_ROOT = BASE / "waveforms"
OUT_DIR = Path("aligned_out"); OUT_DIR.mkdir(exist_ok=True)
DBG_DIR = OUT_DIR / "debug"; DBG_DIR.mkdir(parents=True, exist_ok=True)

MEAS = ["HR","RR","SpO2","Perf","1min_HRV","5min_HRV"]
MODS = ["Pleth"]

CLIP_SECONDS = 30.0
MAX_CSNS = None            # set to an int to cap CSNs; None = no cap
DO_WAVE = True
SAVE_WAVE_CLIPS = False

# Use subset file already filtered to 000–060
SPLIT_FILE = BASE / "available _csns_000_060.csv"
# =========================================

def log(msg): 
    print(f"[mcmed] {msg}")

def safe_to_datetime(s): 
    return pd.to_datetime(s, utc=True, errors="coerce")

def save_head(df, name, n=10):
    p = DBG_DIR / f"{name}_head.csv"
    df.head(n).to_csv(p, index=False)
    log(f"saved head({n}) → {p}")

def load_visits():
    cols = ["CSN","Age","Gender"]
    v = pd.read_csv(BASE/"visits.csv", usecols=cols).dropna(subset=["CSN"])
    v["CSN"] = v["CSN"].astype(np.int64)

    # normalize gender
    gmap = {"f":"female","female":"female","m":"male","male":"male"}
    v["Gender"] = v["Gender"].astype(str).str.strip().str.lower().map(gmap).fillna("unknown")

    G = pd.get_dummies(v["Gender"], prefix="gender", dtype=np.float32)
    for col in ["gender_female","gender_male","gender_unknown"]:
        if col not in G: 
            G[col] = 0.0
    v = pd.concat([v[["CSN","Age"]], G], axis=1)
    v["Age"] = pd.to_numeric(v["Age"], errors="coerce").clip(0, 120).astype(np.float32)

    log(f"visits.csv: {len(v):,} rows, {v['CSN'].nunique():,} CSNs")
    save_head(v, "visits")
    return v

def load_numerics_subset():
    df = pd.read_csv(BASE/"numerics.csv", usecols=["CSN","Source","Measure","Value","Time"])
    df = df[df["Measure"].isin(MEAS)]
    df["Time"] = safe_to_datetime(df["Time"])
    df = df.dropna(subset=["CSN","Time","Value"])
    df["CSN"] = df["CSN"].astype(np.int64)
    df["hour"] = df["Time"].dt.floor("H")
    log(f"numerics.csv → {len(df):,} rows, {df['CSN'].nunique():,} CSNs")

    agg = df.groupby(["CSN","hour","Measure"])["Value"].mean().reset_index()
    wide = agg.pivot(index=["CSN","hour"], columns="Measure", values="Value").reset_index()
    save_head(wide, "numerics_hourly")
    return wide

def csn_wave_dir(csn): 
    return WAVE_ROOT / f"{int(csn)%1000:03d}" / str(int(csn))

def try_import_wfdb():
    try:
        import wfdb  # noqa: F401
        return True
    except Exception:
        return False

def list_segments_for_mod(csn, modality, wfdb):
    base = csn_wave_dir(csn) / modality
    if not base.exists(): 
        return []
    segs = []
    for hea in sorted(base.glob("*.hea")):
        try:
            rec = wfdb.rdrecord(str(hea.with_suffix("")))
            fs, n = float(rec.fs), int(rec.sig_len)
            t0 = datetime.combine(rec.base_date, rec.base_time).replace(tzinfo=timezone.utc)
            t1 = t0 + timedelta(seconds=n/fs)
            segs.append({"path": hea.with_suffix(""), "fs": fs, "n": n, "t0": t0, "t1": t1})
        except Exception as e:
            log(f"[warn] cannot read header {hea}: {e}")
    return segs

def gather_overlap(segs, win_start, win_end):
    overlaps = [s for s in segs if min(s["t1"], win_end) > max(s["t0"], win_start)]
    if not overlaps: 
        return None
    mid = win_start + (win_end - win_start)/2
    # pick segment whose center is closest to the window center
    overlaps.sort(key=lambda s: abs((s["t0"]+(s["t1"]-s["t0"])/2 - mid).total_seconds()))
    s = overlaps[0]
    fs, n = s["fs"], s["n"]
    need = min(int(CLIP_SECONDS * fs), n)
    samp_center = int(((mid - s["t0"]).total_seconds()) * fs)
    samp_from = max(0, min(n - need, samp_center - need//2))
    return {"path": s["path"], "fs": fs, "sampfrom": samp_from, "sampto": samp_from + need}

def attach_wave_pointers(df_hours, wfdb):
    out = df_hours.copy()

    # Initialize parquet-friendly dtypes (nullable)
    for m in MODS:
        out[f"{m}_path"]     = pd.Series(pd.NA, index=out.index, dtype="string")
        out[f"{m}_fs"]       = pd.Series(pd.NA, index=out.index, dtype="Float64")
        out[f"{m}_sampfrom"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out[f"{m}_sampto"]   = pd.Series(pd.NA, index=out.index, dtype="Int64")

    hit_rows = 0
    for csn, g in out.groupby("CSN"):
        segs_by_mod = {m: list_segments_for_mod(csn, m, wfdb) for m in MODS}
        if not any(segs_by_mod.values()):
            continue
        for i in g.index:
            h0 = out.at[i, "hour"]
            h1 = h0 + pd.Timedelta(hours=1)
            for m, segs in segs_by_mod.items():
                if not segs:
                    continue
                pick = gather_overlap(segs, h0.to_pydatetime(), h1.to_pydatetime())
                if pick:
                    # cast to parquet-friendly types
                    out.at[i, f"{m}_path"]     = str(pick["path"])
                    out.at[i, f"{m}_fs"]       = float(pick["fs"])
                    out.at[i, f"{m}_sampfrom"] = int(pick["sampfrom"])
                    out.at[i, f"{m}_sampto"]   = int(pick["sampto"])
                    hit_rows += 1

    log(f"wave pointers filled {hit_rows:,} rows")
    save_head(out, "with_wave_ptrs")
    return out

def main():
    log("start")
    visits = load_visits()
    numer = load_numerics_subset()

    # read subset CSNs 
    if SPLIT_FILE.exists():
        csns = pd.read_csv(SPLIT_FILE, header=None)[0].astype(np.int64).tolist()
        log(f"subset file: {len(csns):,} CSNs listed")
    else:
        log(f"[warn] split file not found: {SPLIT_FILE} → using all numerics CSNs")
        csns = numer["CSN"].unique().astype(np.int64).tolist()

    # apply cap if requested
    if MAX_CSNS is not None:
        csns = csns[:int(MAX_CSNS)]
        log(f"MAX_CSNS cap applied: {len(csns):,} CSNs")

    numer = numer[numer["CSN"].isin(csns)]
    visits = visits[visits["CSN"].isin(csns)]
    log(f"after subset filter: numer={len(numer):,}, visits={len(visits):,}")

    # hourly reindex + ffill per CSN
    feat_cols = [c for c in MEAS if c in numer.columns]
    filled = []
    for csn, g in numer.groupby("CSN"):
        g = g.sort_values("hour").set_index("hour").asfreq("1H")
        g["CSN"] = csn
        g[feat_cols] = g[feat_cols].ffill()
        filled.append(g.reset_index())
    numer_ff = pd.concat(filled, ignore_index=True)
    log(f"after ffill: {len(numer_ff):,} rows")

    # z-normalize with global stats 
    means = numer_ff[feat_cols].mean()
    stds  = numer_ff[feat_cols].std().replace(0, 1.0)
    numer_z = numer_ff.copy()
    numer_z[feat_cols] = (numer_ff[feat_cols] - means) / stds
    save_head(numer_z, "numer_z")

    # add statics
    aligned = numer_z.merge(visits, on="CSN", how="left")
    log(f"after merge: {len(aligned):,} rows")

    # optional waveform pointers
    if DO_WAVE and try_import_wfdb():
        import wfdb  
        aligned = attach_wave_pointers(aligned, wfdb)
    else:
        log("wfdb not available or DO_WAVE=False → skipping wave alignment")

    # -------- final parquet-safe dtypes --------
    path_cols  = [c for c in aligned.columns if c.endswith("_path")]
    float_cols = [c for c in aligned.columns if c.endswith("_fs")]
    from_cols  = [c for c in aligned.columns if c.endswith("_sampfrom")]
    to_cols    = [c for c in aligned.columns if c.endswith("_sampto")]

    for c in path_cols:
        aligned[c] = aligned[c].apply(lambda x: str(x) if pd.notna(x) else pd.NA).astype("string")
    for c in float_cols:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce").astype("Float64")
    for c in (from_cols + to_cols):
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce").astype("Int64")

    if path_cols or float_cols or from_cols or to_cols:
        log("[dtypes] wave columns before parquet:")
        log(aligned[path_cols + float_cols + from_cols + to_cols].dtypes.to_string())

    # write parquet
    out_parquet = OUT_DIR / "aligned_hours.parquet"
    aligned.to_parquet(out_parquet, index=False)
    log(f"wrote {out_parquet} (rows={len(aligned):,}, cols={aligned.shape[1]})")
    log("done")

if __name__ == "__main__":
    main()
