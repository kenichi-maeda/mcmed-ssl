from mcmed_dataset import PatientDataset
import numpy as np
import pandas as pd

ds = PatientDataset(
    base_dir="physionet.org/files/mc-med/1.0.1/data",
    split="train",
    task="ssl",
    min_seq_len=8,
    max_seq_len=8,
    eval_seq_len=8,
    length=10,
)

print("n candidates:", len(ds.csns))
print("first 5 csns:", ds.csns[:5])
print("len ds", len(ds))

sample = ds[0]
print("signals_timeseries1 mean:", sample["signals_timeseries1"].mean())
print("signals_timeseries2 mean:", sample["signals_timeseries2"].mean())

print("\n=== Sample summary ===")
for k, v in sample.items():
    if isinstance(v, np.ndarray):
        print(f"{k:25s} shape={v.shape} dtype={v.dtype}")
    else:
        print(f"{k:25s}", type(v), v)

# --- Peek actual content ---
print("\n=== Peek into values ===")

# Show a few structured vitals
print("structured_timeseries1 (first 2 timesteps):")
print(pd.DataFrame(sample["structured_timeseries1"][:2],
                   columns=["HR","RR","SBP","DBP","MAP","SpO2","Temp"]))

# Show waveform snippet (first 50 samples of first minute)
sig = sample["signals_timeseries1"][0, 0, :50]
print("\nwaveform (first 50 samples):", np.round(sig, 3))

# Statics check
print("\nstatics1:", sample["statics1"])
print("end_idx:", sample["end_idx"])
print("pt_id:", sample["pt_ids"])
