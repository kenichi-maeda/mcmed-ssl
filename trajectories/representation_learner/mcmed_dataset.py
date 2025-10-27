# mcmed_dataset.py

import os, random
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil import parser
import wfdb
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")


# ----------------------------
# Small utilities
# ----------------------------
def _safe_to_py_datetime(x):
    """Parse MC-MED times into *python* datetime (avoid pandas nanosecond overflow)."""
    if isinstance(x, str):
        return parser.isoparse(x)
    return x.to_pydatetime() if hasattr(x, "to_pydatetime") else x

def _minute_grid_from_numerics(numerics_df, csn, measures, ffill_limit=5, bfill_limit=5):
    """Pivot per-CSN numerics to a 1-min grid with columns = measures."""
    df = numerics_df[numerics_df.CSN == csn].copy()
    if df.empty:
        raise ValueError(f"No numerics for CSN={csn}")
    df["Time_py"] = df["Time"].map(_safe_to_py_datetime)
    df = df.sort_values("Time_py")
    t0 = df["Time_py"].iloc[0]
    df["minute"] = df["Time_py"].apply(lambda t: int((t - t0).total_seconds() // 60))

    wide = df.pivot_table(index="minute", columns="Measure", values="Value", aggfunc="mean")
    if wide.empty:
        raise ValueError(f"Empty grid for CSN={csn}")

    full_idx = pd.RangeIndex(wide.index.min(), wide.index.max() + 1)
    grid = wide.reindex(full_idx).ffill(limit=ffill_limit).bfill(limit=bfill_limit)

    # ensure all measures exist (add NaN columns if missing)
    for m in measures:
        if m not in grid.columns:
            grid[m] = np.nan
    grid = grid[list(measures)].astype(float)

    # fill any remaining NaNs by column means (like the example placeholder)
    vals = grid.values
    col_means = np.nanmean(vals, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)  # if a whole column is NaN → 0.0
    rr, cc = np.where(np.isnan(vals))
    if len(rr):
        vals[rr, cc] = col_means[cc]
        grid.iloc[:, :] = vals

    # python datetimes per minute
    times_py = [t0 + timedelta(minutes=int(m)) for m in grid.index]
    return grid, times_py

def _wave_root_for_csn(base_wave_dir, csn):
    """MC-MED waveforms are bucketed by the last 3 digits."""
    suffix = f"{int(csn) % 1000:03d}"
    root = Path(base_wave_dir) / suffix / str(int(csn))
    return root if root.exists() else None

def _list_wfdb_records(base_wave_dir, csn):
    root = _wave_root_for_csn(base_wave_dir, csn)
    if root is None:
        return []
    return sorted([p.with_suffix('') for p in root.glob("**/*.hea")])

def _pick_preferred_record(recs, prefer_dirs=("II", "Pleth", "Resp")):
    """Return one WFDB record path, preferring ECG II > Pleth > Resp if present."""
    by_dir = {r.parts[-2]: r for r in recs if len(r.parts) >= 2}
    for name in prefer_dirs:
        if name in by_dir:
            return by_dir[name]
    return recs[0] if recs else None # for now, only XXXX_01.hat 

def _wfdb_load_1d(base_wave_dir, csn, prefer_dirs=("II","Pleth","Resp")):
    """Load one WFDB record (1 channel) as (arr, fs, start_dt)."""
    recs = _list_wfdb_records(base_wave_dir, csn)
    if not recs:
        return None, None, None
    rec = _pick_preferred_record(recs, prefer_dirs)
    if rec is None:
        return None, None, None

    # Read header & samples
    h = wfdb.rdheader(str(rec))
    sig, fields = wfdb.rdsamp(str(rec))
    fs = float(fields["fs"])
    names = fields["sig_name"]

    # Prefer "II", else first channel
    pref = [["II","ECG"], ["PLETH","Pleth","PPG"], ["RESP","Resp"]]
    upper = [s.upper() for s in names]
    ch_idx = 0
    for grp in pref:
        for alias in grp:
            if alias.upper() in upper:
                ch_idx = upper.index(alias.upper()); break
        else:
            continue
        break

    arr = sig[:, ch_idx].astype(np.float32)

    # Build a python datetime for record start if available
    if h.base_date is not None and h.base_time is not None:
        d = pd.Timestamp(h.base_date)
        t = h.base_time if not isinstance(h.base_time, str) else pd.Timestamp(h.base_time).time()
        start_dt = pd.Timestamp.combine(d, t).tz_localize("UTC").to_pydatetime()
    else:
        start_dt = None

    return arr, fs, start_dt

def _extract_10s_clip(arr, fs, center_time, rec_start_dt, target_sr=125, seconds=10):
    """Take last 10 seconds ending at center_time. Resample to target_sr with linear interp."""
    target_n = target_sr * seconds
    if arr is None or fs is None:
        return np.zeros(target_n, dtype=np.float32)

    if rec_start_dt is None:
        end_idx = len(arr)                      # fallback: take last 10 s of the file
    else:
        dt_sec = max(0.0, (center_time - rec_start_dt).total_seconds())
        end_idx = int(round(dt_sec * fs))
    start_idx = max(0, end_idx - int(round(seconds * fs)))
    y = arr[start_idx:end_idx]
    if len(y) == 0:
        return np.zeros(target_n, dtype=np.float32)

    # resample to exactly target_n points (linear interpolation)
    x_old = np.arange(len(y), dtype=np.float32)
    x_new = np.linspace(0, len(y) - 1, target_n, dtype=np.float32)
    return np.interp(x_new, x_old, y).astype(np.float32)

# ----------------------------
# Dataset — API & augs match example_dataset.PatientDataset
# ----------------------------
class PatientDataset(Dataset):
    """MC-MED-backed replacement that preserves SMD-SSL's PatientDataset API/augs."""
    def __init__(
        self,
        min_seq_len:   int = 8,
        max_seq_len:   int = 8,
        eval_seq_len:  int = 8,
        seed:          int = 0,
        task:          str = "ssl",        # 'ssl' or a downstream task name (e.g., 'example_task')
        signal_seconds:int = 10,
        signal_mask:   float = 0.25,
        history_cutout_prob: float = 0.8,
        history_cutout_frac: float = 0.5,
        spatial_dropout_rate: float = 0.1,
        corrupt_rate: float = 0.6,

        # MC-MED specifics (paths/split/measures)
        base_dir: str = "./physionet.org/files/mc-med/1.0.1/data",
        split:    str = "train",           # 'train' | 'val' | 'test'
        measures: tuple = ("HR","RR","SBP","DBP","MAP","SpO2","Temp"),
        min_rows_per_csn: int = 60,
        length: int = 20000,               # virtual length (windows per epoch)
        prefer_channels: tuple = ("Pleth",), # ("II","Pleth","Resp")
        target_sr: int = 125
    ):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.eval_seq_len = eval_seq_len
        self.seed = seed
        self.task = task
        self.signal_seconds = signal_seconds
        self.signal_mask = signal_mask
        self.history_cutout_prob = history_cutout_prob
        self.history_cutout_frac = history_cutout_frac
        self.spatial_dropout_rate = spatial_dropout_rate
        self.corrupt_rate = corrupt_rate

        self.base_dir = base_dir
        self.wave_dir = os.path.join(base_dir, "waveforms")
        self.split = split
        self.measures = measures
        self.min_rows_per_csn = min_rows_per_csn
        self.length = length
        self.prefer_channels = prefer_channels
        self.target_sr = target_sr

        # --- load split & tables ---
        split_file = {
            "train": "split_random_train.csv",
            "val":   "split_random_val.csv",
            "test":  "split_random_test.csv",
        }[split]
        self.spl = pd.read_csv(os.path.join(base_dir, split_file), header=None, names=["CSN"]).astype({"CSN":"int64"})
        self.visits   = pd.read_csv(os.path.join(base_dir, "visits.csv"))
        self.numerics = pd.read_csv(os.path.join(base_dir, "numerics.csv"))

        # ensure dtypes
        for df in (self.visits, self.numerics):
            df["CSN"] = df["CSN"].astype("int64")

        # --- select candidate CSNs (has waveform & enough numerics) ---
        counts = self.numerics.groupby("CSN")["Time"].count()
        rich = set(counts[counts >= self.min_rows_per_csn].index.tolist())

        # # check waveform existence by folder
        # csns = []
        # for csn in self.spl["CSN"].tolist():
        #     root = _wave_root_for_csn(self.wave_dir, csn)
        #     if csn in rich and root is not None and any(root.glob("**/*.hea")):
        #         csns.append(int(csn))
        # if not csns:
        #     # fall back to numerics-only if nothing found
        #     csns = [int(c) for c in self.spl["CSN"].tolist() if int(c) in rich]
        # self.csns = csns

        # build set of CSNs that actually have WFDB headers locally
        wave_csns = set()
        wave_root = Path(self.wave_dir)
        # only scan buckets that exist locally (you said you have 000–030)
        for bucket in sorted(p.name for p in wave_root.iterdir() if p.is_dir()):
            # skip buckets outside your local subset if needed:
            if not bucket.isdigit() or not (0 <= int(bucket) <= 30):
                continue
            for csn_dir in (wave_root / bucket).iterdir():
                if not csn_dir.is_dir():
                    continue
                # check any *.hea under this csn folder
                if list(csn_dir.rglob("*.hea")):
                    wave_csns.add(int(csn_dir.name))

        split_csns = set(int(c) for c in self.spl["CSN"].tolist())
        # keep those that are: in split ∩ numerics-rich ∩ have waveforms locally
        csns = list(split_csns & rich & wave_csns)

        if not csns:
            # fall back to numerics-only rich CSNs (you’ll get zero waveforms then)
            csns = [int(c) for c in self.spl["CSN"].tolist() if int(c) in rich]

        self.csns = csns


        # RNG
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self):
        # like the placeholder (fixed virtual length), but you can set = len(self.csns) if preferred
        return self.length

    def subsample(self, frac, seed=0):
        rng = np.random.RandomState(seed)
        k = max(1, int(round(len(self.csns) * float(frac))))
        self.csns = rng.choice(self.csns, size=k, replace=False).tolist()

    # ----------------------------
    # Encoders (same shapes as example)
    # ----------------------------
    def make_signals_traj(self, csn, times):
        """Return [T, 1, 125*signal_seconds] extracted around each minute in `times`."""
        arr, fs, start_dt = _wfdb_load_1d(self.wave_dir, csn, self.prefer_channels)
        T = len(times)
        S = self.target_sr * self.signal_seconds

        # after: arr, fs, start_dt = _wfdb_load_1d(...)
        # if arr is not None and start_dt is not None:
        #     rec_len_sec = len(arr) / fs
        #     rec_start = start_dt
        #     rec_end   = start_dt + pd.to_timedelta(rec_len_sec, unit="s")
            # print("WF span:", rec_start, "→", rec_end)
            # print("Window:", times[0], "→", times[-1])

        if arr is None:
            return np.zeros((T, 1, S), dtype=np.float32)
        clips = [
            _extract_10s_clip(arr, fs, t, start_dt, target_sr=self.target_sr, seconds=self.signal_seconds)
            for t in times
        ]

        norm_clips = []
        for y in clips:
            # y shape: (S,)
            m = float(np.median(y))
            s = float(np.std(y))
            if s < 1e-6:            # flat signal → avoid NaN/Inf
                y = y - m
            else:
                y = (y - m) / s
            y = np.clip(y, -5.0, 5.0)  # optional, tame spikes
            norm_clips.append(y.astype(np.float32))

        return np.stack(norm_clips, axis=0)[:, None, :]  # [T, 1, S]
        # return np.stack(clips, axis=0)[:, None, :]  # [T, 1, S]

    def make_structured_data_traj(self, csn, times):
        """Return [T, F] per-minute numerics (already normalized by grid building)."""
        grid, all_times = _minute_grid_from_numerics(self.numerics, csn, self.measures)
        # map requested minute-stamps to rows (we aligned times from the same grid build)
        # here, `times` already comes from that grid, so just slice by position.
        # (callers pass a consecutive sub-window of the grid)
        # For safety, rebuild indices from the same grid:
        t0 = all_times[0]
        idxs = [int((t - t0).total_seconds() // 60) for t in times]
        base_idx = grid.index.min()
        locs = [i - base_idx for i in idxs]
        return grid.iloc[locs].to_numpy(dtype=np.float32)

    def make_static_data(self, csn):
        """Return a small static vector (e.g., age + gender one-hot), like example."""
        row = self.visits.loc[self.visits["CSN"] == csn]
        if row.empty:
            return np.zeros(3, dtype=np.float32)
        row = row.iloc[0]
        age = float(row.get("Age", np.nan))
        g = str(row.get("Gender", ""))
        gender_f = 1.0 if g == "F" else 0.0
        gender_m = 1.0 if g == "M" else 0.0
        age = 0.0 if np.isnan(age) else age
        return np.array([age, gender_f, gender_m], dtype=np.float32)

    # ----------------------------
    # Augmentations (ported 1:1 from example_dataset.py)
    # ----------------------------
    def mask_augmentation(self, signal):
        # Signal: [T, C, S]
        crop_rate = self.signal_mask
        T, C, S = signal.shape
        if crop_rate == 0:
            return signal, np.zeros(T)
        for t in range(T):
            for c in range(C):
                crop_len = int(crop_rate * S)
                crop_start = np.random.randint(0, S) if S > 0 else 0
                stdval = 0.5
                noise = 0.5 * stdval * np.random.randn(crop_len).astype(np.float32)
                if crop_start + crop_len <= S:
                    signal[t, c, crop_start:crop_start + crop_len] = noise
                else:
                    remainder = crop_len - (S - crop_start)
                    signal[t, c, crop_start:S] = noise[:S - crop_start]
                    if remainder > 0:
                        signal[t, c, 0:remainder] = noise[S - crop_start:]
        return signal, None

    def struct_timeseries_augment(self, data):
        # data: [T, F] — cutout then channel dropout; add small noise
        crop_prob = self.history_cutout_prob
        crop_frac = self.history_cutout_frac
        T, F = data.shape
        masked = np.zeros_like(data, dtype=np.float32)

        if crop_prob != 0:
            for c in range(F):
                if np.random.uniform() < crop_prob:
                    crop_len = max(1, int(crop_frac * T))
                    crop_start = np.random.randint(0, max(1, T - crop_len + 1))
                    data[crop_start:crop_start + crop_len, c] = np.nan
                    masked[crop_start:crop_start + crop_len, c] = 1
            # forward/backward fill like example
            data = pd.DataFrame(data).ffill().values
            data = pd.DataFrame(data).bfill().values

        # add noise
        for c in range(F):
            data[:, c] += np.random.normal(size=T).astype(np.float32) * 0.1

        # channel dropout
        for c in range(F):
            if np.random.uniform() < self.spatial_dropout_rate:
                data[:, c] = 0.0
                masked[:, c] = 1
        return data.astype(np.float32), masked

    def statics_augment(self, data):
        # data: [F_static], corrupt subset & add noise
        masked = np.zeros_like(data, dtype=np.float32)
        num_feat = len(data)
        num_corrupt = int(self.corrupt_rate * num_feat)
        if num_corrupt > 0:
            corrupt_idxs = np.random.choice(num_feat, size=num_corrupt, replace=False)
            for i in corrupt_idxs:
                data[i] = 0.0
                masked[i] = 1.0
        for i in range(num_feat):
            data[i] += np.random.normal() * 0.1
        return data.astype(np.float32), masked
    

    # ----------------------------
    # __getitem__ — same keys & shapes as the original
    # ----------------------------
    def __getitem__(self, idx):
        # pick a CSN (round-robin)
        csn = self.csns[idx % len(self.csns)]
        # build the 1-min grid once so we can choose a window & reuse for both views
        grid, times_py = _minute_grid_from_numerics(self.numerics, csn, self.measures)

        # if self.task == "ssl":
        #     # random contiguous window length in [min_seq_len, max_seq_len]
        #     start = np.random.randint(0, max(1, len(grid) - self.min_seq_len + 1))
        #     traj_len = np.random.randint(self.min_seq_len, self.max_seq_len + 1)
        #     end = min(start + traj_len, len(grid))
        #     times = times_py[start:end]
        # else:
        #     # fine-tune/eval: fixed eval_seq_len from the start
        #     start = 0
        #     end = min(start + self.eval_seq_len, len(grid))
        #     times = times_py[start:end]

        # 2) Load waveform once to know its time span
        arr, fs, start_dt = _wfdb_load_1d(self.wave_dir, csn, self.prefer_channels)

        def wf_span():
            if arr is None or fs is None or start_dt is None:
                return None, None
            dur = len(arr) / fs
            return start_dt, start_dt + pd.to_timedelta(dur, unit="s")

        rec_start, rec_end = wf_span()

        # 3) Choose a window of timestamps ‘times’ that overlaps WF span
        if self.task == "ssl":
            need = np.random.randint(self.min_seq_len, self.max_seq_len + 1)
        else:
            need = self.eval_seq_len

        # “valid” minutes are those whose 10s tail fits inside the recording
        def valid(t):
            if rec_start is None:  # no waveform → allow any (will become zeros)
                return True
            # keep a tiny epsilon and require t <= rec_end
            return (t > rec_start) and (t <= rec_end)

        valid_idx = [i for i, t in enumerate(times_py) if valid(t)]

        def pick_consecutive(indices, need):
            """Pick a consecutive block of length 'need' from 'indices', if possible."""
            if not indices:
                return None
            runs = []
            run = [indices[0]]
            for a, b in zip(indices, indices[1:]):
                if b == a + 1:
                    run.append(b)
                else:
                    if len(run) >= need: runs.append(run)
                    run = [b]
            if len(run) >= need: runs.append(run)
            if not runs:
                return None
            run = random.choice(runs)
            start_pos = random.randint(0, len(run) - need)
            return run[start_pos:start_pos + need]

        # try to pick a valid overlapping window first
        sel = pick_consecutive(valid_idx, need)

        if sel is None:
            # fallback: no overlap long enough → pick any consecutive window from the grid
            # (signals will be zeros for out-of-span minutes; structured still fine)
            if len(grid) >= need:
                j0 = np.random.randint(0, len(grid) - need + 1)
                sel = list(range(j0, j0 + need))
            else:
                # shorter than need → take all and pad later
                sel = list(range(len(grid)))

        # 4) Build the final per-minute timestamps
        times = [times_py[i] for i in sel]

        # core arrays
        signals = self.make_signals_traj(csn, times)                 # [T, 1, S]
        structured = self.make_structured_data_traj(csn, times)      # [T, F]
        statics = self.make_static_data(csn)                          # [F_static]
        T = len(times)

        # end_idx = index of last real timestep (before any padding)
        end_idx = int(T - 1)

        # Pad to max_seq_len/eval_seq_len like the original behavior if needed
        target_T = (self.max_seq_len if self.task == "ssl" else self.eval_seq_len)
        if T < target_T:
            pad_T = target_T - T
            # pad signals by repeating last frame
            if T > 0:
                last_sig = signals[-1:]
                signals = np.concatenate([signals, np.repeat(last_sig, pad_T, axis=0)], axis=0)
                last_struct = structured[-1:]
                structured = np.concatenate([structured, np.repeat(last_struct, pad_T, axis=0)], axis=0)
            else:
                S = self.target_sr * self.signal_seconds
                signals = np.zeros((target_T, 1, S), dtype=np.float32)
                structured = np.zeros((target_T, structured.shape[1]), dtype=np.float32)


        # Augmentations & outputs
        if self.task == "ssl":
            # Two *independent* views per modality (same as example_dataset)
            sig1, _ = self.mask_augmentation(signals.copy())
            sig2, _ = self.mask_augmentation(signals.copy())

            st1, _ = self.struct_timeseries_augment(structured.copy())
            st2, _ = self.struct_timeseries_augment(structured.copy())

            lab1, _ = self.statics_augment(statics.copy())
            lab2, _ = self.statics_augment(statics.copy())

            return {
                    # base, unaugmented (needed by MetaModel/run loop)
                    "signals_timeseries":    signals.astype(np.float32),
                    "structured_timeseries": structured.astype(np.float32),
                    "statics":               statics.astype(np.float32),
                    "end_idx":               np.array(end_idx, dtype=np.int64),
                    "pt_ids":                str(csn),
                    "start_times":           times[0].timestamp() if len(times) else -1.0,

                    # SSL views (two augmentations per modality)
                    "signals_timeseries1":   sig1.astype(np.float32),
                    "signals_timeseries2":   sig2.astype(np.float32),
                    "structured_timeseries1":st1.astype(np.float32),
                    "structured_timeseries2":st2.astype(np.float32),
                    "statics1":              lab1.astype(np.float32),
                    "statics2":              lab2.astype(np.float32),
                }
        else:
            # Supervised fine-tuning
            row = self.visits.loc[self.visits["CSN"] == csn]
            if not row.empty:
                dispo = str(row.iloc[0].get("DC_dispo", "")).upper()
                if "DECEASED" in dispo or "EXPIRED" in dispo:
                    label = 1
                else:
                    label = 0
            else:
                label = 0  # fallback

            out = {
                "signals_timeseries": signals.astype(np.float32),
                "structured_timeseries": structured.astype(np.float32),
                "statics": statics.astype(np.float32),
                "end_idx": np.array(end_idx, dtype=np.int64),
                "pt_ids": str(csn),
                "start_times": times[0].timestamp() if len(times) else -1.0,
                "example_task": np.array(label, dtype=np.int64),
            }
            return out
