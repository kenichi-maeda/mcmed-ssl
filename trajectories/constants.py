import os
from pathlib import Path

PROJECT_NAME                = 'trajectories'

PROJECTS_BASE = os.environ.get('PROJECTS_BASE')
if not PROJECTS_BASE:
    PROJECTS_BASE = str(Path(__file__).resolve().parents[1])

PROJECT_DIR                 = os.path.join(PROJECTS_BASE, PROJECT_NAME)
RUNS_DIR                    = os.path.join(PROJECT_DIR, 'runs')
ARGS_FILENAME               = 'args.json'
FINE_TUNE_ARGS_FILENAME     = 'fine_tune_args.json'

ALL_TASKS = [
    "ed_dispo",
]


TASK_OUTPUT_DIMENSIONS = {
    "ed_dispo": 2,
}

TRAIN, TUNING, HELD_OUT = 'train', 'tuning', 'held out'