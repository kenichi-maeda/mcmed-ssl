import argparse, json, pickle
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, asdict

from ..constants import *
class BaseArgs(ABC):
    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath, mode='r') as f: return cls(**json.loads(f.read()))
    @staticmethod
    def from_pickle_file(filepath):
        with open(filepath, mode='rb') as f: return pickle.load(f)

    def to_dict(self): return asdict(self)
    def to_json_file(self, filepath):
        with open(filepath, mode='w') as f: f.write(json.dumps(asdict(self), indent=4))
    def to_pickle_file(self, filepath):
        with open(filepath, mode='wb') as f: pickle.dump(self, f)

    @classmethod
    @abstractmethod
    def _build_argparse_spec(cls, parser):
        raise NotImplementedError("Must overwrite in base class!")

    @classmethod
    def from_commandline(cls):
        parser = argparse.ArgumentParser()

        # To load from a run_directory (not synced to overall structure above):
        parser.add_argument(
            "--do_load_from_dir", action='store_true',
            help="Should the system reload from the sentinel args.json file in the specified run directory "
                 "(--run_dir) and use those args rather than consider those set here? If so, no other args "
                 "need be set (they will all be ignored).",
            default=False
        )

        main_dir_arg, args_filename = cls._build_argparse_spec(parser)

        args = parser.parse_args()

        if args.do_load_from_dir:
            load_dir = vars(args)[main_dir_arg]
            assert os.path.exists(load_dir), "Dir (%s) must exist!" % load_dir
            args_path = os.path.join(load_dir, args_filename)
            assert os.path.exists(args_path), "Args file (%s) must exist!" % args_path

            print('PATH TO ARGS: ', args_path)

            new_args = cls.from_json_file(args_path)
            print(new_args)

            assert os.path.samefile(vars(new_args)[main_dir_arg], load_dir),\
                f"{main_dir_arg}: {vars(new_args)[main_dir_arg]} doesn't match loaded file: {load_dir}!"

            return new_args

        args_dict = vars(args)
        if 'do_load_from_dir' in args_dict: args_dict.pop('do_load_from_dir')

        return cls(**args_dict)

def intlt(bounds):
    start, end = bounds if type(bounds) is tuple else (0, bounds)
    def fntr(x):
        x = int(x)
        if x < start or x >= end: raise ValueError("%d must be in [%d, %d)" % (x, start, end))
        return x
    return fntr

def within(s):
    def fntr(x):
        if x not in s: raise ValueError("%s must be in {%s}!" % (x, ', '.join(s)))
        return x
    return fntr

@dataclass
class Args(BaseArgs):
    # Configuration (do not change)
    max_seq_len:             int   = 10
    min_seq_len:             int   = 8
    eval_seq_len:            int   = 8

    # Run Params (set)
    modeltype:               str   = "cnn_gru"
    run_dir:                 str   = "./output"
    run_name:                str   = "test"
    model_file_template:     str   = "model" # can use {arg} format syntax
    do_overwrite:            bool  = False # should overwrite run dir?
    dataset_dir:             str   = None # Not used by default--inferred from rotation.
    num_dataloader_workers:  int   = 4 # Num dataloader workers. Can increase.

    # Training Params (set)
    epochs:                  int   = 15
    do_train:                bool  = True
    do_eval_train:           bool  = True
    do_eval_tuning:          bool  = True
    train_save_every:        int   = 1
    batches_per_gradient:    int   = 1
    signal_seconds:          int   = 10
    
    # Modality choices
    only_ecg:                  bool = False
    only_tabular:              bool = False

    # SSL parameters
    do_simclr:               bool = False
    do_vicreg:               bool = False
    spatial_dropout_rate:    float = 0.1
    history_cutout_frac:     float = 0.5
    history_cutout_prob:     float = 0.5
    expander_fcs:            tuple = (128,128)
    simclr_temp:             float = 0.1
    vicreg_lambda:           float = 25.0
    vicreg_mu:               float = 25.0
    detach:                  bool  = False   ## Whether to detach component reprs before passing into RNN.
    signal_mask:             float = 0.25
    global_weight:           float = 1.0
    component_weight:            float = 1.0
    component_only_epochs:       int   = -1   # Train signal encoder ONLY at beginning or not. 
        
    ## Args related to tabular, static SSL
    corrupt_rate:            float = 0.6

    # Hyperparameters (tune)
    batch_size:              int   = 32
    learning_rate:           float = 1e-4
    # We track this separately to give hyperparameter tuning an easier way to disable this.
    do_learning_rate_decay:  bool  = True
    learning_rate_decay:     float = 1 # decay gamma. 1 is no change.
    learning_rate_step:      int   = 1
    gru_num_hidden:          int   = 2
    cnn_enc_dim:             int   = 512
    gru_pooling_method:      str   = 'last'
    do_bidirectional:        bool  = False
    fc_layer_sizes:          tuple = (256,)
    # We track this separately to give hyperparameter tuning an easier way to disable this.
    do_weight_decay:         bool  = True
    weight_decay:            float = 0
    tab_enc_dim:             int = 128 # output dim of tabular encoder.

    frac_data:               float= 1.0 # how much of the fine_tuning data should we use?
    frac_data_seed:          int  = 0 

    # Debug
    do_test_run:             bool  = False
    do_detect_anomaly:       bool  = False

    # Dataset control (virtual length = windows per epoch)
    train_windows_per_epoch: int = 20000

    # --- Dataset I/O (SSL/pretraining) ---
    parquet_path:        str = "../aligned_out/aligned_hours.parquet"
    use_splits_train:    str = "split_random_train_subset.csv"
    use_splits_val:      str = "split_random_val_subset.csv"
    use_splits_test:     str = "split_random_test_subset.csv"   # not used in SSL usually, but handy to have
    modality:            str = "Pleth"
    ssl_stride:          int = 8                          # non-overlapping 8-hr windows for SSL


    @classmethod
    def _build_argparse_spec(cls, parser):
        # Configuration (do not change)
        parser.add_argument("--max_seq_len", type=int, default=10, help="maximum number of timepoints to feed into the model")
        parser.add_argument("--min_seq_len", type=int, default=8,
                            help="minimum number of timepoints in a sampled window")
        parser.add_argument("--eval_seq_len", type=int, default=8,
                            help="fixed eval window length")

        # Run Params (set)
        parser.add_argument("--run_dir", type=str, required=True, help='save dir.')
        parser.add_argument("--run_name", type=str, required=False, default='test', help='experiment name on wandb')
        
        parser.add_argument("--do_overwrite", action='store_true', default=False, help='Should overwrite existent save_dir?')
        parser.add_argument("--no_do_overwrite", action='store_false', dest='do_overwrite')
        parser.add_argument('--dataset_dir', type=str, default=None, help='Explicit dataset path (else use rotation).')
        parser.add_argument('--num_dataloader_workers', type=int, default=4, help='# dataloader workers.')

        # Training Params (set)
        parser.add_argument("--modeltype", type=str, default='cnn_gru', choices =  ['cnn_gru',], help="model architecture")
        parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
        parser.add_argument("--do_eval_train", action="store_true", help="set flag to train the model", default=True)
        parser.add_argument("--do_eval_tuning", action="store_true", help="set flag to val the model", default=True)
        parser.add_argument('--train_save_every', type=int, default = 1, help='Save the model every ? epochs?')
        parser.add_argument(
            '--batches_per_gradient', type=int, default = 1,
            help='Accumulate gradients over this many batches.'
        )
        parser.add_argument('--signal_seconds', type=int, default = 10, help='How many seconds of signal to use?')
        
        # Which modalities to use?
        parser.add_argument("--only_ecg", action='store_true', default=False, help='only use ECG')
        parser.add_argument("--only_tabular", action='store_true', default=False, help='only use tabular timeseries')

        # SSL parameters
        parser.add_argument('--do_simclr', action='store_true', default=False)
        parser.add_argument('--do_vicreg', action='store_true', default=False)
        parser.add_argument('--detach', action='store_true', default=False)
        parser.add_argument('--spatial_dropout_rate', type=float, default=0.1)
        parser.add_argument('--history_cutout_frac', type=float, default=0.5)
        parser.add_argument('--history_cutout_prob', type=float, default=0.0)
        parser.add_argument('--corrupt_rate', type=float, default=0.6)
        parser.add_argument("--expander_fcs", type=int, nargs='+', default=[128,128], help="expander layers")
        parser.add_argument('--simclr_temp', type=float, default=0.1)
        parser.add_argument('--signal_mask', type=float, default=0.25)
        parser.add_argument('--vicreg_lambda', type=float, default=25.0)
        parser.add_argument('--vicreg_mu', type=float, default=25.0)
        parser.add_argument('--global_weight', type=float, default=1.0)
        parser.add_argument('--component_weight', type=float, default=1.0)
        parser.add_argument('--component_only_epochs', type=int, default=-1)

        # Hyperparameters (tune)
        parser.add_argument('--weight_decay', type=float, default=0, help="L2 weight decay penalty")
        parser.add_argument('--do_weight_decay', action='store_true', default=True, help="Do L2 weight decay?")
        parser.add_argument("--batch_size", type=int, default=32, help="batch size for train, test, and eval")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for the model")
        parser.add_argument('--do_learning_rate_decay', action='store_true', default=True, help="Do learning rate decay?")
        parser.add_argument('--no_do_learning_rate_decay', dest='do_learning_rate_decay', action='store_false')
        parser.add_argument("--learning_rate_decay", type=float, default=1, help="lr decay factor")
        parser.add_argument("--learning_rate_step", type=int, default=1, help="#epochs / lr decay")
        parser.add_argument("--gru_num_hidden", type=int, default=2, help="Number of hidden layers for GRU")
        parser.add_argument("--cnn_enc_dim", type=int, default=512, help="CNN encoder dimension.")
        parser.add_argument("--gru_pooling_method", type=str, default='last', help="GRU pooling style.")
        parser.add_argument("--tab_enc_dim", type=int, default=128, help="Tabular feature encoder output layer size.")

        parser.add_argument('--do_bidirectional', action='store_true', default=True, help='bidirectional?')
        parser.add_argument('--fc_layer_sizes', type=int, nargs='+', default=(256,), help='cnn fc stack')
        parser.add_argument('--frac_data', type=float, default=1.0, help='# dataloader workers.')

        parser.add_argument('--parquet_path', type=str, required=True,
                    help='Path to the hourly-aligned parquet (one row per visit-hour).')
        parser.add_argument('--use_splits_train', type=str, required=True,
                            help='CSV of CSNs for train split (one CSN per line).')
        parser.add_argument('--use_splits_val', type=str, required=True,
                            help='CSV of CSNs for val/tuning split.')
        parser.add_argument('--use_splits_test', type=str, default=None,
                            help='CSV of CSNs for test split (optional for SSL).')
        parser.add_argument('--modality', type=str, default='Pleth',
                            choices=['Pleth'], help='Signal modality to use (Pleth wired).')
        parser.add_argument('--ssl_stride', type=int, default=8,
                            help='Step between starting hours when building SSL windows (8 = non-overlap).')


        # Debug
        parser.add_argument('--do_test_run', action='store_true', default=False, help='Will use small dataset. Faster runtime.')
        parser.add_argument('--do_detect_anomaly', action='store_true', default=False, help='Will detect nans. Slower runtime.')

        parser.add_argument('--train_windows_per_epoch', type=int, default=20000,
                    help='Number of training windows per epoch (virtual length).')

        return 'run_dir', ARGS_FILENAME


@dataclass
class FineTuneArgs(BaseArgs):
    run_dir:                  str   = "" # required
    run_name:                 str   = "test"    
    num_dataloader_workers:   int   = 4 # Num dataloader workers. Can increase.
    frac_fine_tune_data_seed: int   = 0 # how much of the fine_tuning data should we use?
    train_embedding_after:    int   = -1 # should the embedding be frozen (-1) or trained after a number of epochs?
    do_frozen_representation: bool  = False # linear evaluation
    do_free_representation:   bool  = True # Full finetuning eval
    do_small_data:            bool  = False # Whether to also train across various small-data levels
    verbose:                  bool  = False
    signal_seconds:           int   = 10
    load_epoch:               int   = -1
    epochs:                   int = 10

    # Hyperparameters
    batch_size:               int   = 32
    learning_rate:            float = 1e-4

    # SSL
    do_simclr:                bool = False
    do_vicreg:                bool = False
    
    # Task
    task:                   str = 'example_task'

    # --- Dataset I/O (FT) ---
    parquet_path:        str = "aligned_out/aligned_hours.parquet"
    use_splits_train:    str = "split_random_train_subset.csv"
    use_splits_val:      str = "split_random_val_subset.csv"
    use_splits_test:     str = "split_random_test_subset.csv"
    modality:            str = "Pleth"

    # Window lengths for FT (often same as SSL)
    min_seq_len:         int = 8
    max_seq_len:         int = 8
    eval_seq_len:        int = 8

    # --- Normalization for FT structured/statics ---
    norm_json:                 str = "./norm_stats.json"  # where to save/load struct stats from train
    standardize_structured:    bool = True
    standardize_statics:       bool = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument(
            "--task", type=within(ALL_TASKS), help="Which task?"
        )
        parser.add_argument('--do_small_data', action='store_true', default=True, help='Small Data?')
        parser.add_argument('--do_frozen_representation', action='store_true', default=True, help='FTD?')
        parser.add_argument('--do_free_representation', action='store_true', default=False, help='FTE?')
        parser.add_argument("--run_dir", type=str, required=True, help="Dir for this generalizability exp.")
        parser.add_argument("--run_name", type=str, required=False, default='test', help='experiment name on wandb')
        parser.add_argument('--num_dataloader_workers', type=int, default=4, help='# dataloader workers.')
        parser.add_argument('--frac_fine_tune_data_seed', type=int, default=0, help='random seed for subsampling the data for fine_tuning')
        parser.add_argument('--signal_seconds', type=int, default = 10, help='How many seconds of signal to use?')
        parser.add_argument('--load_epoch', type=int, default = -1, help='What epoch to load from? -1 implies latest.')
        parser.add_argument('--epochs', type=int, default = 10, help='How many epochs to train for.')

        parser.add_argument("--batch_size", type=int, default=32, help="batch size for train, test, and eval")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for the model")

        parser.add_argument('--train_embedding_after', type=int, default=-1, help='Decide whether the embedding should be frozen (-1) or trained after this number of epochs. An argument of 0 will train the embedding for the whole fine-tuning window.')
        parser.add_argument('--verbose', action='store_true', help='print to motality file')
        # SSL parameters
        parser.add_argument('--do_simclr', action='store_true', default=False)
        parser.add_argument('--do_vicreg', action='store_true', default=False)

        # Dataset I/O
        parser.add_argument('--parquet_path', type=str, required=True,
                            help='Path to the hourly-aligned parquet.')
        parser.add_argument('--use_splits_train', type=str, required=True,
                            help='CSV of CSNs for FT train split.')
        parser.add_argument('--use_splits_val', type=str, required=True,
                            help='CSV of CSNs for FT val split.')
        parser.add_argument('--use_splits_test', type=str, required=True,
                            help='CSV of CSNs for FT test split.')
        parser.add_argument('--modality', type=str, default='Pleth',
                            choices=['Pleth'], help='Signal modality to use.')

        # Window lengths
        parser.add_argument('--min_seq_len', type=int, default=8)
        parser.add_argument('--max_seq_len', type=int, default=8)
        parser.add_argument('--eval_seq_len', type=int, default=8)

        # Normalization
        parser.add_argument('--norm_json', type=str, default='./norm_stats.json',
                            help='Where to save (train) / load (val/test) structured normalization stats.')
        parser.add_argument('--standardize_structured', action='store_true', default=True)
        parser.add_argument('--no_standardize_structured', dest='standardize_structured', action='store_false')
        parser.add_argument('--standardize_statics', action='store_true', default=False)
        parser.add_argument('--no_standardize_statics', dest='standardize_statics', action='store_false')

        return 'run_dir', FINE_TUNE_ARGS_FILENAME