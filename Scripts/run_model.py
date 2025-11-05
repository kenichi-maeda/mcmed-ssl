import os, sys
sys.path.append('../')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tqdm import tqdm

from trajectories.constants import *
from trajectories.representation_learner.args import *
from trajectories.representation_learner.run_model import *


if __name__=="__main__":
    args = Args(
        run_dir="runs/pretrain",
        dataset_dir="physionet.org/files/mc-med/1.0.1/data",
        do_overwrite=True,

        # training
        do_train=True,
        do_vicreg=True,
        do_eval_tuning=True,  
        epochs=15,
        batch_size=128,
        train_windows_per_epoch=500,

        # windowing
        min_seq_len=8, max_seq_len=8,
        eval_seq_len=8,
        ssl_stride=8,              # <-- non-overlap for SSL

        # signals/augs
        signal_seconds=10,
        signal_mask=0.25,
        history_cutout_prob=0.25,
        history_cutout_frac=0.25,
        spatial_dropout_rate=0.0,
        corrupt_rate=0.6,

        # dataloaders
        num_dataloader_workers=8,

        # point the dataset to your aligned parquet and split CSVs
        parquet_path="aligned_out/aligned_hours.parquet",
        use_splits_train="split_random_train_subset.csv",
        use_splits_val="split_random_val_subset.csv"
        
    )
    run_main(args, tqdm=tqdm)