# mcmed_dataset.py
import os, warnings, math, random, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import wfdb
    WFDB_OK = True
except Exception as e:
    WFDB_OK = False
    warnings.warn(f"[PatientDataset] wfdb not available: {e}")

# ---------------------------
# Utility helpers
# ---------------------------
def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _to_utc(t):
    return pd.to_datetime(t, utc=True, errors="coerce")

def _center_crop_1d(x, need):
    """Center-crop a 1D array to length `need`. If x too short, return None."""
    if x is None or len(x) < need:
        return None
    start = (len(x) - need) // 2
    return x[start:start+need]

def _zeros_signal(nsamp):
    return np.zeros((1, nsamp), dtype=np.float32)  # (C=1, S)



# =========================================================
# Custom PatientDataset for MC-MED
# =========================================================
class PatientDataset(Dataset):
    """
    Minimal working MC-MED dataset:
      - consumes an hourly-aligned parquet (one row per visit-hour)
      - pulls 30s Pleth per hour when available, splits into two 10s views
      - builds 8-hour trajectories (min/max_seq_len)
      - augmentations are no-ops (identity)
      - handles NaNs by FFILL per CSN then mean-impute (done offline in your parquet;
        here we also guard at runtime).

    Output dict keys match your original scaffolding:
      - 'signals_timeseries1' : (T, C=1, S=10s*125)
      - 'signals_timeseries2' : (T, C=1, S=10s*125)
      - 'structured_timeseries1' : (T, M)
      - 'structured_timeseries2' : (T, M)
      - 'statics1' : (D,)
      - 'statics2' : (D,)
      - 'signals_timeseries' : alias of signals_timeseries1 (for compatibility)
      - 'pt_ids' : str(CSN)
      - 'start_times' : first hour (as pandas.Timestamp)
      - 'end_idx' : T-1
      - (optional) 'example_task' label if task == 'example_task'
    """

    def __init__(
        self,
        min_seq_len:   int = 8,
        max_seq_len:   int = 8,
        eval_seq_len:  int = 8,
        seed:          int = 0,
        task:          str = 'ssl',
        signal_seconds:int = 10, 
        signal_mask:   float = 0.25, 
        history_cutout_prob: float = 0.25,
        history_cutout_frac: float = 0.25, 
        spatial_dropout_rate: float= 0.0, 
        corrupt_rate: float= 0.6,       
        # --- new args ---
        parquet_path: str = "aligned_mcmed_pleth.parquet",
        modality: str = "Pleth",      # we wire Pleth only in this version
        use_splits: str | None = None,# e.g., "split_random_train.csv" 
        base_dir: str = "physionet.org/files/mc-med/1.0.1/data",
        verbose: bool = True,

        split: str = "train",                 # "train" | "val" | "test"
        norm_json: str | None = None,         # where to save/load stats
        standardize_structured: bool = True,  # apply (W - mean)/std to struct cols
        standardize_statics: bool = False,    # usually keep statics raw; set True if desired
        save_norm_stats: bool = True,         # only effective for split=="train"

        label_source_csv: str | None = "visits.csv",
        label_task: str = "ed_dispo",
        label_positive: tuple = ("admit", "icu", "observation"),
        label_negative: tuple = ("discharge",),
        label_column_candidates: tuple = ("ED_dispo", "ed_dispo", "Disposition", "ED_Disposition"),
        ssl_stride: int = 1
    ):
        super().__init__()
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

        self.modality = modality
        self.verbose = verbose
        self.base_dir = Path(base_dir)

        self.split = split
        self.norm_json = Path(norm_json) if norm_json else None
        self.standardize_structured = bool(standardize_structured)
        self.standardize_statics = bool(standardize_statics)
        self.save_norm_stats = bool(save_norm_stats)

        self.label_source_csv = label_source_csv
        self.label_task = label_task
        self.label_positive = tuple(s.lower() for s in label_positive)
        self.label_negative = tuple(s.lower() for s in label_negative)
        self.label_column_candidates = label_column_candidates

        self.ssl_stride = ssl_stride

        rng = np.random.RandomState(seed)
        self._rng = rng

        # -------------------------
        # 1) Load the aligned parquet
        # -------------------------
        if self.verbose:
            print(f"[PatientDataset] Loading parquet: {parquet_path}")
        self.df = pd.read_parquet(parquet_path)

        # Normalize time dtype
        self.df["hour"] = _to_utc(self.df["hour"])

        # 2) Optional split filter (supports absolute or relative path)
        if use_splits:
            split_path = Path(use_splits)
            if not split_path.is_absolute():
                split_path = self.base_dir / split_path
            if self.verbose:
                print(f"[PatientDataset] Applying split filter: {split_path}")
            try:
                split_ids = pd.read_csv(split_path, header=None)[0].astype(int).tolist()
                before = self.df["CSN"].nunique()
                self.df = self.df[self.df["CSN"].isin(split_ids)]
                after = self.df["CSN"].nunique()
                if self.verbose:
                    print(f"[PatientDataset] CSNs before split: {before} → after: {after}")
            except Exception as e:
                warnings.warn(f"[PatientDataset] Could not apply split filter: {e}")

        # 3) Pick structured + static columns
        self.struct_cols = [c for c in ["HR","RR","SpO2","Perf","1min_HRV","5min_HRV"] if c in self.df.columns]
        self.static_cols = [c for c in ["Age","gender_female","gender_male","gender_unknown"] if c in self.df.columns]
        if self.verbose:
            print(f"[PatientDataset] Structured columns: {self.struct_cols}")
            print(f"[PatientDataset] Static columns: {self.static_cols}")

        # Statics: mean/zero fill (safe)
        if self.static_cols:
            self.df[self.static_cols] = (
                self.df[self.static_cols]
                .fillna(self.df[self.static_cols].mean(numeric_only=True))
                .fillna(0.0)
            )

        # 4) If fine-tuning, load labels and filter to labelable CSNs
        self.csn2label = None
        if self.task == "ed_dispo":
            self.csn2label = self._load_ed_dispo_labels()
            labelable = set(self.csn2label.keys())
            before_rows = len(self.df)
            self.df = self.df[self.df["CSN"].isin(labelable)]
            if self.verbose:
                print(f"[PatientDataset] Labelable CSNs: {len(labelable):,}. "
                    f"Rows before label filter: {before_rows:,} → after: {len(self.df):,}")

        # 5) Compute normalization stats **after** split/label filtering 
        # Structured stats
        if self.struct_cols:
            if self.standardize_structured:
                if self.split == "train":
                    # compute on TRAIN rows only (already filtered by use_splits)
                    train_means = self.df[self.struct_cols].mean(numeric_only=True)
                    train_stds  = self.df[self.struct_cols].std(numeric_only=True).replace(0, 1.0)
                    self.struct_mean = train_means
                    self.struct_std  = train_stds
                    if self.norm_json and self.save_norm_stats:
                        self._save_norm_json(self.norm_json, train_means, train_stds, which="structured")
                        if self.verbose:
                            print(f"[PatientDataset] Saved train normalization to {self.norm_json}")
                else:
                    if self.norm_json and self.norm_json.exists():
                        _, mean, std = self._load_norm_json(self.norm_json)
                        # align to current columns (in case)
                        mean = mean.reindex(self.struct_cols).fillna(0.0)
                        std  = std.reindex(self.struct_cols).fillna(1.0).replace(0, 1.0)
                        self.struct_mean = mean
                        self.struct_std  = std
                        if self.verbose:
                            print(f"[PatientDataset] Loaded normalization from {self.norm_json}")
                    else:
                        # Fallback (leakage!) – compute on this split and warn
                        self.struct_mean = self.df[self.struct_cols].mean(numeric_only=True)
                        self.struct_std  = self.df[self.struct_cols].std(numeric_only=True).replace(0, 1.0)
                        warnings.warn(
                            "[PatientDataset] norm_json missing for non-train split. "
                            "Computed stats on this split (risk of leakage)."
                        )
            else:
                # no standardization requested; still keep for aug noise shape
                self.struct_mean = self.df[self.struct_cols].mean(numeric_only=True)
                self.struct_std  = self.df[self.struct_cols].std(numeric_only=True).replace(0, 1.0)
        else:
            self.struct_mean = pd.Series(dtype=float)
            self.struct_std  = pd.Series(dtype=float)

        # Statics: usually we do NOT standardize; but support it if requested
        if self.static_cols:
            if self.standardize_statics:
                if self.split == "train":
                    s_mean = self.df[self.static_cols].mean(numeric_only=True)
                    s_std  = self.df[self.static_cols].std(numeric_only=True).replace(0, 1.0)
                    self.static_mean = s_mean
                    self.static_std  = s_std
                else:
                    # use the same norm_json if you want; or separate file
                    self.static_mean = self.df[self.static_cols].mean(numeric_only=True)
                    self.static_std  = self.df[self.static_cols].std(numeric_only=True).replace(0, 1.0)
            else:
                # keep raw; set sane defaults for aug noise
                self.static_mean = self.df[self.static_cols].mean(numeric_only=True)
                self.static_std  = self.df[self.static_cols].std(numeric_only=True).replace(0, 1.0)
        else:
            self.static_mean = pd.Series(dtype=float)
            self.static_std  = pd.Series(dtype=float)

        # Runtime NaN guard map
        self.struct_means = (self.df[self.struct_cols].mean(numeric_only=True).to_dict()
                             if self.struct_cols else {})


        # 6) Mark hours that actually have waveform pointers (used to guard empty-wave windows)
        self.df["has_wave"] = (
            self.df.get("Pleth_path").notna()
            # & self.df.get("Pleth_sampfrom").notna()
            # & self.df.get("Pleth_sampto").notna()
        )
        self.min_wave_hours_per_window = 1  # set to 0 to disable the guard

        # 7) Build windows (8-hour trajectories) with the wave guard
        self.windows = []
        T_need = self.max_seq_len if self.min_seq_len != self.max_seq_len else self.min_seq_len
        if self.task == "example_task":
            T_need = self.eval_seq_len

        if self.verbose:
            print(f"[PatientDataset] Building windows with T={T_need} (task={self.task})")


        for csn, g in self.df.groupby("CSN"):
            g = g.sort_values("hour").reset_index(drop=True)
            hw = g["has_wave"].to_numpy(dtype=bool)
            step = self.ssl_stride if self.task == "ssl" else 1
            for i in range(0, len(g) - T_need + 1, step):
                j = i + T_need - 1
                if hw[i:j+1].sum() >= self.min_wave_hours_per_window:
                    self.windows.append((int(csn), i, j))

        if self.verbose:
            print(f"[PatientDataset] After split filter: rows={len(self.df):,}, "
                f"unique CSNs={self.df['CSN'].nunique():,}, windows={len(self.windows):,}")
            print(f"[PatientDataset] % hours with Pleth: {100.0 * float(self.df['has_wave'].mean()):.1f}%")

        # 8) Debug peek
        if self.verbose and len(self.df):
            print("[PatientDataset] Sample df head:")
            print(self.df.head(3).to_string())

    # label loader
    def _load_ed_dispo_labels(self):
        """Return dict CSN->0/1 based on visit disposition."""
        if self.label_source_csv is None:
            raise ValueError("label_source_csv must be set for ed_dispo task")

        src = self.base_dir / self.label_source_csv if not str(self.label_source_csv).startswith("/") else Path(self.label_source_csv)
        if self.verbose:
            print(f"[PatientDataset] Loading labels from: {src}")
        vis = pd.read_csv(src)

        # find disposition column
        dispo_col = None
        for c in self.label_column_candidates:
            if c in vis.columns:
                dispo_col = c; break
        if dispo_col is None:
            raise ValueError(f"Could not find disposition column in {src}. "
                             f"Tried: {self.label_column_candidates}")

        vis = vis.dropna(subset=["CSN", dispo_col]).copy()
        vis["CSN"] = vis["CSN"].astype(np.int64)

        # normalize text
        disp = vis[dispo_col].astype(str).str.strip().str.lower()
        y = np.full(len(vis), fill_value=np.nan, dtype=np.float32)
        y[np.isin(disp, self.label_positive)] = 1.0
        y[np.isin(disp, self.label_negative)] = 0.0

        labeled = vis.loc[np.isfinite(y), ["CSN"]].copy()
        labeled["y"] = y[np.isfinite(y)].astype(int)

        # one label per CSN (if duplicates, pick the last one)
        csn2y = labeled.drop_duplicates(subset=["CSN"], keep="last").set_index("CSN")["y"].to_dict()

        if self.verbose:
            from collections import Counter
            counts = Counter(csn2y.values())
            print(f"[PatientDataset] ED_dispo labelable CSNs: {len(csn2y):,} "
                  f"(class balance: {counts})")
        return csn2y


    def __len__(self):
        return len(self.windows)

    def subsample(self, frac, seed=0):
        """Optional: downsample the index for quick dev."""
        if frac is None or frac >= 1.0:
            return
        rng = np.random.RandomState(seed)
        n = max(1, int(len(self.windows) * frac))
        self.windows = rng.choice(self.windows, size=n, replace=False).tolist()
        if self.verbose:
            print(f"[PatientDataset] Subsampled windows to {len(self.windows)}")

    def _save_norm_json(self, path, means, stds, which="structured"):
        payload = {
            "which": which,
            "cols": list(means.index),
            "mean": {k: float(means[k]) for k in means.index},
            "std":  {k: float(stds[k])  for k in stds.index},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _load_norm_json(self, path):
        import json
        with open(path, "r") as f:
            payload = json.load(f)
        cols = payload["cols"]
        mean = pd.Series(payload["mean"]).astype(float)
        std  = pd.Series(payload["std"]).astype(float)
        std = std.replace(0, 1.0)
        return cols, mean, std


    def mask_augmentation(self, signal):
        """
        signal: (T, C=1, S) float32
        Returns:
        aug_signal: (T,1,S)
        mask_t:     (T,) fraction masked per step (for logging)
        """
        if signal.ndim != 3:
            raise ValueError(f"signals must be (T,C,S); got {signal.shape}")
        T, C, S = signal.shape
        aug = signal.copy()
        masked_frac_per_t = np.zeros((T,), dtype=np.float32)

        frac = float(self.signal_mask) 
        if frac <= 0:
            return aug, masked_frac_per_t

        block_len = max(1, int(round(frac * S)))

        for t in range(T):
            for c in range(C):
                x = aug[t, c]
                # Gaussian noise (σ = 0.25 * std) — paper used 0.25
                std = float(np.nanstd(x))
                if not np.isfinite(std) or std == 0:
                    std = 1.0
                aug[t, c] = x + (0.25 * std) * np.random.randn(S).astype(np.float32)

                # Contiguous mask block → set to 0
                start = np.random.randint(0, S - block_len + 1) if S > block_len else 0
                aug[t, c, start:start+block_len] = 0.0

            masked_frac_per_t[t] = block_len / float(S)

        return aug, masked_frac_per_t


    def struct_timeseries_augment(self, data):
        """
        data: (T, M) float32 structured time series (already standardized or raw).
        Returns:
        aug_data: (T, M)
        mask:     (T, M) 1 where corrupted/masked, else 0
        """
        if data.ndim != 2:
            raise ValueError(f"structured data must be (T,M); got {data.shape}")
        T, M = data.shape
        aug = data.copy()
        mask = np.zeros_like(aug, dtype=np.float32)

        if M == 0 or T == 0:
            return aug, mask

        crop_prob = float(self.history_cutout_prob)  # e.g., 0.25 recommended
        crop_frac = float(self.history_cutout_frac)  # e.g., 0.25 recommended
        ch_drop   = float(self.spatial_dropout_rate) # optional channel dropout

        # Build per-feature mean/std arrays for noise & dropout impute
        # If not present, fallback to 0/1
        means = np.array([self.struct_mean.get(self.struct_cols[j], 0.0) if j < len(self.struct_cols) else 0.0
                        for j in range(M)], dtype=np.float32)
        stds  = np.array([self.struct_std.get(self.struct_cols[j], 1.0) if j < len(self.struct_cols) else 1.0
                        for j in range(M)], dtype=np.float32)
        stds[~np.isfinite(stds)] = 1.0
        stds[stds == 0] = 1.0

        # --- history cutout ---
        if crop_prob > 0 and crop_frac > 0:
            cut_len = max(1, int(round(crop_frac * T)))
            for j in range(M):
                if np.random.rand() < crop_prob and T >= cut_len:
                    start = np.random.randint(0, T - cut_len + 1)
                    aug[start:start+cut_len, j] = np.nan
                    mask[start:start+cut_len, j] = 1.0

            # ffill then bfill (short series safe)
            df = pd.DataFrame(aug)
            df = df.ffill().bfill()
            aug = df.to_numpy(dtype=np.float32)

        # --- noise addition (0.1 * std) ---
        noise = np.random.randn(T, M).astype(np.float32) * (0.1 * stds)[None, :]
        aug = aug + noise

        # --- optional channel dropout (impute mean) ---
        if ch_drop > 0:
            for j in range(M):
                if np.random.rand() < ch_drop:
                    aug[:, j] = means[j]
                    mask[:, j] = 1.0

        # final NaN guard
        aug = np.nan_to_num(aug, nan=means, posinf=means, neginf=means).astype(np.float32)

        return aug, mask


    def statics_augment(self, data):
        """
        data: (D,) float32 static features
        Returns:
        aug:  (D,)
        mask: (D,) 1 where corrupted/dropped, else 0
        """
        if data.ndim != 1:
            raise ValueError(f"statics must be (D,); got {data.shape}")
        D = data.shape[0]
        aug = data.copy()
        mask = np.zeros_like(aug, dtype=np.float32)
        if D == 0:
            return aug, mask

        # mean/std vectors aligned to columns in self.static_cols
        means = np.array([self.static_mean.get(self.static_cols[j], 0.0) if j < len(self.static_cols) else 0.0
                        for j in range(D)], dtype=np.float32)
        stds  = np.array([self.static_std.get(self.static_cols[j], 1.0) if j < len(self.static_cols) else 1.0
                        for j in range(D)], dtype=np.float32)
        stds[~np.isfinite(stds)] = 1.0
        stds[stds == 0] = 1.0

        # dropout
        frac = float(self.corrupt_rate)  # e.g., 0.25
        k = int(round(frac * D))
        if k > 0:
            idx = np.random.choice(D, size=k, replace=False)
            aug[idx] = means[idx]
            mask[idx] = 1.0

        # noise
        aug = aug + (0.1 * stds) * np.random.randn(D).astype(np.float32)

        # final NaN guard
        aug = np.nan_to_num(aug, nan=means, posinf=means, neginf=means).astype(np.float32)
        return aug, mask


    def _load_pleth_10s_pair(self, row):
        """
        Load a 30s slice for this hour and split into two 10s clips.
        Returns (sig10_a, sig10_b) each shaped (1, 1250), float32, z-normalized per-clip.
        """
        fs_target = 125
        need_30 = 30 * fs_target
        need_10 = 10 * fs_target
        zeros10 = _zeros_signal(need_10)

        path = row.get("Pleth_path", None)
        if path is None or pd.isna(path) or (isinstance(path, str) and not len(path)):
            return zeros10, zeros10
        if not WFDB_OK:
            warnings.warn("[PatientDataset] wfdb not available; returning zeros")
            return zeros10, zeros10

        sampfrom = row.get("Pleth_sampfrom", np.nan)
        sampto   = row.get("Pleth_sampto",   np.nan)
        row_fs   = _safe_float(row.get("Pleth_fs", fs_target), fs_target)

        def _z(x, eps=1e-6):
            m = np.nanmean(x)
            s = np.nanstd(x)
            if not np.isfinite(m): m = 0.0
            if (not np.isfinite(s)) or (s < eps): s = 1.0
            x = (x - m) / s
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return x

        try:
            # --- read samples ---
            if not np.isnan(sampfrom) and not np.isnan(sampto):
                rec = wfdb.rdrecord(path, sampfrom=int(sampfrom), sampto=int(sampto))
            else:
                rec = wfdb.rdrecord(path)

            x = rec.p_signal if rec.p_signal is not None else rec.d_signal
            if x is None or x.size == 0:
                return zeros10, zeros10

            # Ensure we have a 1-D array (nsamp,) if multiple channels appear, take first
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 2:
                x = x[:, 0]

            # If we loaded the whole record (no sampfrom/sampto), center-crop to ~30s at row_fs
            if np.isnan(sampfrom) or np.isnan(sampto):
                need_row = int(round(30 * row_fs))
                x = _center_crop_1d(x, need_row)
                if x is None:
                    return zeros10, zeros10

            # --- resample / decimate to 125 Hz ---
            if int(round(row_fs)) != fs_target:
                factor = int(round(row_fs / fs_target))
                if factor > 1:
                    # Prefer antialiased decimation if scipy is installed
                    try:
                        from scipy.signal import decimate
                        x = decimate(x, factor, ftype='iir', zero_phase=True).astype(np.float32)
                    except Exception:
                        x = x[::factor]  # fallback
                else:
                    # row_fs < 125: optional upsample to reach target length (simplest: pad/return zeros)
                    # Minimal fallback: if we can’t upsample here, bail gracefully.
                    # (You can implement proper resample_poly if you want true upsampling.)
                    return zeros10, zeros10

            # Ensure at least 30s @125Hz
            x30 = _center_crop_1d(x, need_30)
            if x30 is None:
                return zeros10, zeros10

            # Two disjoint 10s: first and last 10s of the 30s window
            a = x30[:need_10]
            b = x30[-need_10:]

            # Per-clip z-normalization (recommended for Pleth)
            a = _z(a).reshape(1, -1).astype(np.float32)
            b = _z(b).reshape(1, -1).astype(np.float32)
            return a, b

        except Exception as e:
            if self.verbose:
                print(f"[PatientDataset] waveform read error @ {path}: {e}")
            return zeros10, zeros10


    # -------------------------
    # Structured + statics for a slice (rows i..j inclusive)
    # -------------------------
    def _build_struct_and_statics(self, g_slice):
        """
        g_slice: DataFrame for one CSN, rows i..j (T rows)
        Returns:
          W  : (T, M) float32 (structured)
          d  : (D,)   float32 (statics from the *first* row)
        """
        T = len(g_slice)
        # structured
        if self.struct_cols:
            W = g_slice[self.struct_cols].copy()
            # Runtime guard against NaN: fill with precomputed means
            for c in self.struct_cols:
                if W[c].isna().any():
                    W[c] = W[c].fillna(self.struct_means.get(c, 0.0))
            W = W.to_numpy(dtype=np.float32)
        else:
            W = np.zeros((T, 0), dtype=np.float32)

        # statics (take from first row)
        if self.static_cols:
            d = g_slice.iloc[0][self.static_cols].to_numpy(dtype=np.float32)
            # final fallback NaN→0
            if np.isnan(d).any():
                d = np.nan_to_num(d, nan=0.0)
        else:
            d = np.zeros((0,), dtype=np.float32)

        return W, d

    # -------------------------
    # __getitem__
    # -------------------------
    def __getitem__(self, idx):
        csn, i0, i1 = self.windows[idx]
        sub = self.df[self.df["CSN"] == csn].sort_values("hour").reset_index(drop=True)
        g_slice = sub.iloc[i0:i1+1].copy()
        T = len(g_slice)

        if self.verbose and idx < 3:  # only log first few for sanity
            print(f"[PatientDataset][{idx}] CSN={csn} rows={i0}-{i1} (T={T}) hours {g_slice['hour'].iloc[0]} → {g_slice['hour'].iloc[-1]}")

        # -------- signals: build two views --------
        sigs1 = []
        sigs2 = []
        for _, row in g_slice.iterrows():
            a, b = self._load_pleth_10s_pair(row)  # (1, 1250) each
            # Always torch-friendly final format later; keep numpy now
            sigs1.append(a)
            sigs2.append(b)

        # Stack to (T, C=1, S)
        signals_timeseries1 = np.stack(sigs1, axis=0)  # (T,1,1250)
        signals_timeseries2 = np.stack(sigs2, axis=0)  # (T,1,1250)

        if self.task == "ssl" and self.signal_mask > 0:
            sig1_aug, _ = self.mask_augmentation(signals_timeseries1.copy())
            sig2_aug, _ = self.mask_augmentation(signals_timeseries2.copy())
        else:
            sig1_aug, sig2_aug = signals_timeseries1, signals_timeseries2

        # -------- structured & statics  --------
        W, d = self._build_struct_and_statics(g_slice)  # (T,M), (D,)

        # Standardize structured/statics if requested (using train stats)
        if self.standardize_structured and W.size and len(self.struct_cols):
            # vectorized: (T, M) - (M,) / (M,)
            W = (W - self.struct_mean.values[None, :]) / self.struct_std.values[None, :]
        if self.standardize_statics and d.size and len(self.static_cols):
            d = (d - self.static_mean.values) / self.static_std.values


        # Keep identical copies for view2 (since no-ops)
        structured_timeseries1, _ = self.struct_timeseries_augment(W.copy())
        structured_timeseries2, _ = self.struct_timeseries_augment(W.copy())
        statics1, _ = self.statics_augment(d.copy())
        statics2, _ = self.statics_augment(d.copy())

        # -------- pack tensors --------
        tensors = {}
        tensors["signals_timeseries1"] = torch.from_numpy(sig1_aug)  # (T,1,1250)
        tensors["signals_timeseries2"] = torch.from_numpy(sig2_aug)  # (T,1,1250)
        tensors["signals_timeseries"]  = tensors["signals_timeseries1"]  # alias


        tensors["structured_timeseries1"] = torch.from_numpy(structured_timeseries1)  # (T,M)
        tensors["structured_timeseries2"] = torch.from_numpy(structured_timeseries2)  # (T,M)

        tensors["statics1"] = torch.from_numpy(statics1)  # (D,)
        tensors["statics2"] = torch.from_numpy(statics2)  # (D,)

        tensors["structured_timeseries"] = tensors["structured_timeseries1"]
        tensors["statics"] = tensors["statics1"]

        tensors["pt_ids"] = torch.tensor(int(csn), dtype=torch.long)
        t0 = pd.Timestamp(g_slice["hour"].iloc[0])
        tensors["start_times"] = torch.tensor(int(t0.timestamp()), dtype=torch.long)
        tensors["end_idx"] = T - 1  # index of last valid step

        # >>> NEW: attach label for fine-tuning
        if self.task == "ed_dispo":
            y = int(self.csn2label.get(int(csn)))  # guaranteed present after filter
            tensors["ed_dispo"] = torch.tensor(y, dtype=torch.long)

        # quick sanity logs (first few only)
        if self.verbose and idx < 3:
            s1 = tensors["signals_timeseries1"].shape
            sW = tensors["structured_timeseries1"].shape
            sd = tensors["statics1"].shape
            print(f"[PatientDataset][{idx}] shapes: signals={s1}, W={sW}, d={sd}, end_idx={tensors['end_idx']}")

        return tensors


if __name__ == "__main__":
    ds = PatientDataset(
        parquet_path="aligned_out/aligned_hours.parquet",
        modality="Pleth",
        task="ssl",
        min_seq_len=8, max_seq_len=8,
        verbose=True,
        label_source_csv="visits.csv",
        use_splits="split_random_train_subset.csv",
        standardize_structured=False,
        ssl_stride=8
    )

    # ds = PatientDataset(
    #     parquet_path="aligned_out/aligned_hours.parquet",
    #     modality="Pleth",
    #     task="ed_dispo",
    #     min_seq_len=8, max_seq_len=8,
    #     verbose=True,
    #     label_source_csv="visits.csv",
    #     use_splits="split_random_train_subset.csv",
    #     standardize_structured=True, 
    #     norm_json="runs/ft_task/norm_struct.json",
    #     save_norm_stats=True,
    # )

    print("len(ds) =", len(ds))
    sample = ds[0]  # get one example

    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={tuple(v.shape)} | dtype={v.dtype}")
            # show first few values for numeric tensors
            arr = v.detach().cpu().numpy().flatten()
            print("  sample values:", np.round(arr[:8], 4))
        else:
            print(f"{k}: {v}")
