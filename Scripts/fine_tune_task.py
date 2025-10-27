import sys
sys.path.append('../')

import pickle
from tqdm import tqdm

from trajectories.constants import *
from trajectories.representation_learner.args import FineTuneArgs
from trajectories.representation_learner.fine_tune import *


if __name__=="__main__":
    ft_args = FineTuneArgs(
        run_dir="../runs/pretrain",
        task="example_task",

        # choose one:
        do_frozen_representation=False,   # linear probe (head only, encoder frozen)
        do_free_representation=True,    # set True to fine-tune encoder

        # training hyperparams for the FT run
        epochs=5,
        batch_size=64,
        learning_rate=1e-4,

        # which checkpoint to start from
        load_epoch=4,     # use 4 to load runs/pretrain/model.epoch-4 (use -1 for latest)

        # unfreeze timing for full FT; for frozen probe keep -1
        train_embedding_after=-1,

        # how much raw waveform to use (passed through to dataset)
        signal_seconds=10,
    )
    main(ft_args, tqdm=tqdm)