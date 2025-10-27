import sys
sys.path.append('../')

import pickle
from tqdm import tqdm

from trajectories.constants import *
from trajectories.representation_learner.args import FineTuneArgs
from trajectories.representation_learner.lineval import *


if __name__=="__main__":
    ft_args = FineTuneArgs(
        run_dir="../runs/pretrain",
        load_epoch=4,                      # loads ../runs/pretrain/model.epoch-4

        task="example_task",

        do_frozen_representation=True,
        do_free_representation=False,

        # data / train hparams for representation extraction and sklearn head
        batch_size=64,
        epochs=1,                          # epochs here don’t train the encoder (frozen),
                                           # they are still read by helpers; keep small
        learning_rate=1e-4,                # unused by sklearn, but kept for API compatibility
        num_dataloader_workers=4,
        signal_seconds=10,    
        verbose=False,
    )
    main(ft_args, tqdm=tqdm)