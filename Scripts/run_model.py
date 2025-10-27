import sys
sys.path.append('../')

from tqdm import tqdm

from trajectories.constants import *
from trajectories.representation_learner.args import *
from trajectories.representation_learner.run_model import *


if __name__=="__main__":
    args = Args(
        run_dir="../runs/pretrain",
        dataset_dir="../physionet.org/files/mc-med/1.0.1/data",
        do_train=True,
        do_vicreg=True,
        do_eval_tuning=True,  
        epochs=5,
        batch_size=64,
        train_windows_per_epoch=1000,
        do_overwrite=True
    )
    main(args, tqdm=tqdm)