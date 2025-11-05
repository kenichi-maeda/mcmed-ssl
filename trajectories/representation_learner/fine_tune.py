"""
fine_tune.py
Fine tunes a pre-trained model on a specific (single) task
"""

import torch
import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from .mcmed_dataset import PatientDataset

import numpy as np

import json, os, pickle, random
from copy import deepcopy
from tqdm import tqdm
import glob
import wandb

from ..utils import *
from ..constants import *
from .args import Args

from .meta_model import *
from .run_model import setup_datasets_and_dataloaders, train_meta_model, evaluate_model

def build_ft_datasets(pretrain_run_dir, base_dir, parquet_path):
    # paths for MC-MED official splits
    split_map = {
        "train": "split_random_train.csv",
        "val":   "split_random_val.csv",
        "test":  "split_random_test.csv",
    }

    # where we’ll save/load structured normalization for FT
    ft_root = os.path.join(pretrain_run_dir, "ed_dispo", "FTF")  # matches below
    os.makedirs(ft_root, exist_ok=True)
    norm_path = os.path.join(ft_root, "norm_struct.json")

    ds_train = PatientDataset(
        task="ed_dispo",
        parquet_path=parquet_path,
        base_dir=base_dir,
        use_splits=split_map["train"],
        split="train",
        # labels
        label_source_csv="visits.csv",
        # structured normalization: compute+save on train
        standardize_structured=True,
        norm_json=norm_path,
        save_norm_stats=True,
        # signals
        modality="Pleth",
        signal_seconds=10,
        verbose=False,
    )
    # reuse same stats for val/test (no saving here)
    ds_val = PatientDataset(
        task="ed_dispo",
        parquet_path=parquet_path,
        base_dir=base_dir,
        use_splits=split_map["val"],
        split="val",
        label_source_csv="visits.csv",
        standardize_structured=True,
        norm_json=norm_path,
        save_norm_stats=False,
        modality="Pleth",
        signal_seconds=10,
        verbose=False,
    )
    ds_test = PatientDataset(
        task="ed_dispo",
        parquet_path=parquet_path,
        base_dir=base_dir,
        use_splits=split_map["test"],
        split="test",
        label_source_csv="visits.csv",
        standardize_structured=True,
        norm_json=norm_path,
        save_norm_stats=False,
        modality="Pleth",
        signal_seconds=10,
        verbose=False,
    )
    return ds_train, ds_val, ds_test

def build_ft_datasets(pretrain_run_dir, base_dir, parquet_path):
    # paths for MC-MED official splits
    split_map = {
        "train": "split_random_train_subset.csv",
        "val":   "split_random_val_subset.csv",
        "test":  "split_random_test_subset.csv",
    }

    # where we’ll save/load structured normalization for FT
    ft_root = os.path.join(pretrain_run_dir, "ed_dispo", "FTF")  # matches below
    os.makedirs(ft_root, exist_ok=True)
    norm_path = os.path.join(ft_root, "norm_struct.json")

    ds_train = PatientDataset(
        task="ed_dispo",
        parquet_path=parquet_path,
        base_dir=base_dir,
        use_splits=split_map["train"],
        split="train",
        # labels
        label_source_csv="visits.csv",
        # structured normalization: compute+save on train
        standardize_structured=True,
        norm_json=norm_path,
        save_norm_stats=True,
        # signals
        modality="Pleth",
        signal_seconds=10,
        verbose=False,
    )
    # reuse same stats for val/test (no saving here)
    ds_val = PatientDataset(
        task="ed_dispo",
        parquet_path=parquet_path,
        base_dir=base_dir,
        use_splits=split_map["val"],
        split="val",
        label_source_csv="visits.csv",
        standardize_structured=True,
        norm_json=norm_path,
        save_norm_stats=False,
        modality="Pleth",
        signal_seconds=10,
        verbose=False,
    )
    ds_test = PatientDataset(
        task="ed_dispo",
        parquet_path=parquet_path,
        base_dir=base_dir,
        use_splits=split_map["test"],
        split="test",
        label_source_csv="visits.csv",
        standardize_structured=True,
        norm_json=norm_path,
        save_norm_stats=False,
        modality="Pleth",
        signal_seconds=10,
        verbose=False,
    )
    return ds_train, ds_val, ds_test

def ft_collate(batch):
    # default_collate for tensor fields, drop metadata fields that break collation
    keep = {
        "signals_timeseries1","signals_timeseries2","signals_timeseries",
        "structured_timeseries1","structured_timeseries2",
        "statics1","statics2","ed_dispo"
    }
    out = {k: [] for k in keep}
    end_idx_list = []
    for b in batch:
        for k in list(b.keys()):
            if k in keep:
                out[k].append(b[k])
        end_idx_list.append(b["end_idx"])
    # stack tensors
    for k in out:
        out[k] = torch.stack(out[k], dim=0)

    out["structured_timeseries"] = out["structured_timeseries1"]
    out["statics"] = out["statics1"]
    out["end_idx"] = torch.tensor(end_idx_list, dtype=torch.long)
    return out


def fine_tune_model(
    fine_tune_args, meta_model_args, sample_datum, train_dataloaders_by_data_frac,
    tqdm=None, meta_model=None, tuning_dataloader=None, test_dataloader=None,
):
    print('Finetuning model...')
    reloaded = (meta_model is not None)

    verbose = False
    if hasattr(fine_tune_args, 'verbose'):
        verbose = fine_tune_args.verbose

    # Decide on the task weights so that we only train/eval on the chosen task
    task_weights = {i:0.0 for i in ALL_TASKS}
    task_weights[fine_tune_args.task] = 1.0
    
    outputs = []
    for data_frac, train_dataloader in train_dataloaders_by_data_frac.items():
        wandb_exp = fine_tune_args.run_dir.split('/')[-1]
        if data_frac != 1:
            wandb_name = wandb_exp + '_FT_' + fine_tune_args.task + f'_frac{data_frac}'
        else:
            wandb_name = wandb_exp + '_FT_' + fine_tune_args.task

        # wandb.init(project = "traj", 
        #            entity="trajectories",
        #            config = vars(fine_tune_args),
        #            name = wandb_name, 
        #            tags=["FT", wandb_exp, fine_tune_args.task],
        #            resume=False,
        #            settings=wandb.Settings(start_method="fork"))
        wandb_state = 'FT'
        
        
        fine_tune_dir_name = fine_tune_args.task
        if data_frac != 1: fine_tune_dir_name += f"_{str(data_frac).replace('.', '-')}"

        fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)
        assert os.path.isdir(fine_tune_run_dir), f"{fine_tune_run_dir} must exist!"

        if meta_model is None:
            meta_model = MetaModel(
                meta_model_args, sample_datum,
                class_names = None,
                task_weights = task_weights,
                verbose = verbose,
            )

        load_epoch = fine_tune_args.load_epoch
        if load_epoch == -1:
            load_epoch='latest'
        if not(reloaded):
            reloaded, epoch = meta_model.load(epoch=load_epoch)
            epoch=0
            reloaded=False

        if fine_tune_args.do_frozen_representation:
            meta_model_FTD = meta_model
            meta_model_FTD_args = deepcopy(meta_model_args)

            meta_model_FTD.run_dir = os.path.join(fine_tune_run_dir, "FTD")
            meta_model_FTD.freeze_representation()
            meta_model_FTD_args.run_dir = meta_model_FTD.run_dir
            if not os.path.isdir(meta_model_FTD.run_dir): os.makedirs(meta_model_FTD.run_dir)

            best_model = train_meta_model(
                meta_model_FTD, train_dataloader, meta_model_FTD_args, reloaded=reloaded, epoch=epoch,
                tuning_dataloader=tuning_dataloader, tqdm=tqdm,
                train_embedding_after=fine_tune_args.train_embedding_after, wandb=wandb, wandb_state=wandb_state
            )
            outputs.append(meta_model_FTD)
            eval_results = evaluate_model(best_model, tuning_dataloader, test_dataloader)

            # --- save metrics & preds for FTD ---
            save_dir = meta_model_FTD_args.run_dir
            TASK = fine_tune_args.task
            val_auc = eval_results["val"].get(f"{TASK}_auc")
            val_auprc = eval_results["val"].get(f"{TASK}_auprc")
            test_auc = eval_results["test"].get(f"{TASK}_auc")
            test_auprc = eval_results["test"].get(f"{TASK}_auprc")
            ft_metrics = {
                "strategy": "FTD",
                "val_auc": float(val_auc) if val_auc is not None else None,
                "val_auprc": float(val_auprc) if val_auprc is not None else None,
                "test_auc": float(test_auc) if test_auc is not None else None,
                "test_auprc": float(test_auprc) if test_auprc is not None else None,
            }
            with open(os.path.join(save_dir, "ft_metrics.json"), "w") as f:
                json.dump(ft_metrics, f, indent=2)
            if "example_task_logit" in eval_results["test"] and "example_task_label" in eval_results["test"]:
                test_logits = np.concatenate(eval_results["test"]["example_task_logit"], axis=0)
                test_labels = np.concatenate(eval_results["test"]["example_task_label"], axis=0)
                np.save(os.path.join(save_dir, "ft_best_preds_labels.npy"), [test_logits[:,1], test_labels])

            torch.save(eval_results, os.path.join(save_dir, 'eval_metrics.pt'))
            meta_model = None  # reset so the next branch re-loads cleanly if needed

        if fine_tune_args.do_free_representation:
            meta_model_FTF = meta_model
            meta_model_FTF_args = deepcopy(meta_model_args)

            meta_model_FTF.run_dir = os.path.join(fine_tune_run_dir, "FTF")
            meta_model_FTF_args.run_dir = meta_model_FTF.run_dir
            if not os.path.isdir(meta_model_FTF.run_dir): os.makedirs(meta_model_FTF.run_dir)

            best_model = train_meta_model(
                meta_model_FTF, train_dataloader, meta_model_FTF_args, reloaded=reloaded, epoch=epoch,
                tuning_dataloader=tuning_dataloader, tqdm=tqdm,
                train_embedding_after=fine_tune_args.train_embedding_after, wandb=wandb, wandb_state=wandb_state
            )
            outputs.append(meta_model_FTF)
            eval_results = evaluate_model(best_model, tuning_dataloader, test_dataloader)

            # --- save metrics & preds for FTF ---
            save_dir = meta_model_FTF_args.run_dir
            TASK = fine_tune_args.task
            val_auc = eval_results["val"].get(f"{TASK}_auc")
            val_auprc = eval_results["val"].get(f"{TASK}_auprc")
            test_auc = eval_results["test"].get(f"{TASK}_auc")
            test_auprc = eval_results["test"].get(f"{TASK}_auprc")
            ft_metrics = {
                "strategy": "FTF",
                "val_auc": float(val_auc) if val_auc is not None else None,
                "val_auprc": float(val_auprc) if val_auprc is not None else None,
                "test_auc": float(test_auc) if test_auc is not None else None,
                "test_auprc": float(test_auprc) if test_auprc is not None else None,
            }
            with open(os.path.join(save_dir, "ft_metrics.json"), "w") as f:
                json.dump(ft_metrics, f, indent=2)
            if "example_task_logit" in eval_results["test"] and "example_task_label" in eval_results["test"]:
                test_logits = np.concatenate(eval_results["test"]["example_task_logit"], axis=0)
                test_labels = np.concatenate(eval_results["test"]["example_task_label"], axis=0)
                np.save(os.path.join(save_dir, "ft_best_preds_labels.npy"), [test_logits[:,1], test_labels])

            torch.save(eval_results, os.path.join(save_dir, 'eval_metrics.pt'))
            meta_model = None

        # for k,v in eval_results.items(): 
        #     for metric_name, metric_value in v.items(): 
        #         wandb.log({metric_name+'_'+k : metric_value}) # metric is a single number   
    return outputs

def main(fine_tune_args, tqdm):
    
    ### SEED EVERYTHING HERE ###
    random.seed(fine_tune_args.frac_fine_tune_data_seed)
    torch.manual_seed(fine_tune_args.frac_fine_tune_data_seed)
    np.random.seed(fine_tune_args.frac_fine_tune_data_seed)

    assert os.path.isdir(fine_tune_args.run_dir), "Run dir must exist!"
    assert (
        fine_tune_args.do_frozen_representation or
        fine_tune_args.do_free_representation
    ), "Need to do either FTF or FTD!"

    fine_tune_args.to_json_file(os.path.join(fine_tune_args.run_dir, FINE_TUNE_ARGS_FILENAME))

    assert fine_tune_args.task in ALL_TASKS,\
        f"Invalid fine tune task: {fine_tune_args.task}"

    meta_model_args = Args.from_json_file(os.path.join(fine_tune_args.run_dir, ARGS_FILENAME))
    
    meta_model_args.run_dir = fine_tune_args.run_dir
    assert os.path.isdir(meta_model_args.run_dir), meta_model_args.run_dir

    print('Disabling contrastive learning for fine tuning!')
    meta_model_args.do_simclr = False
    meta_model_args.do_vicreg = False

    print('Loading LR, batch size, epochs from the FT args!')
    meta_model_args.learning_rate = fine_tune_args.learning_rate
    meta_model_args.batch_size = fine_tune_args.batch_size
    meta_model_args.epochs = fine_tune_args.epochs
    meta_model_args.signal_seconds = fine_tune_args.signal_seconds

    BASE_DIR   = "physionet.org/files/mc-med/1.0.1/data"
    PARQUET    = "aligned_out/aligned_hours.parquet"
        
    # datasets, train_dataloader, val_dataloader, test_dataloader = setup_datasets_and_dataloaders(meta_model_args, 
    # assert datasets['train'].max_seq_len == meta_model_args.max_seq_len
    # assert train_dataloader.dataset.max_seq_len == meta_model_args.max_seq_len
    # sample_datum = datasets['train'][0]
                      
    ds_train, ds_val, ds_test = build_ft_datasets(
        pretrain_run_dir=fine_tune_args.run_dir,
        base_dir=BASE_DIR,
        parquet_path=PARQUET,
    )

    train_dataloader = DataLoader(
        ds_train, batch_size=meta_model_args.batch_size,
        num_workers=meta_model_args.num_dataloader_workers,
        pin_memory=True, shuffle=True, collate_fn=ft_collate
    )
    val_dataloader = DataLoader(
        ds_val, batch_size=meta_model_args.batch_size,
        num_workers=meta_model_args.num_dataloader_workers,
        pin_memory=True, shuffle=False, collate_fn=ft_collate
    )
    test_dataloader = DataLoader(
        ds_test, batch_size=meta_model_args.batch_size,
        num_workers=meta_model_args.num_dataloader_workers,
        pin_memory=True, shuffle=False, collate_fn=ft_collate
    )                                                                    

    sample_datum = ds_train[0] 

    # NOTE: this could be extended to support dataloaders of different size
    train_dataloaders_by_data_frac = {1: train_dataloader}

    fine_tune_dir_name = fine_tune_args.task

    fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)

    if not os.path.exists(fine_tune_run_dir): os.makedirs(fine_tune_run_dir)

    data_frac_seed = random.randint(0, int(1e10))
    with open(os.path.join(fine_tune_run_dir, 'data_frac_seed.txt'), mode='w') as f:
        f.write(str(data_frac_seed))

    for (do, suffix) in [(fine_tune_args.do_frozen_representation, "FTD"), 
                            (fine_tune_args.do_free_representation, "FTF")]:
        if not do: 
            continue

        fine_tune_meta_model_args = deepcopy(meta_model_args)
        fine_tune_meta_model_args.run_dir = os.path.join(fine_tune_run_dir, suffix)

        if not os.path.exists(fine_tune_meta_model_args.run_dir): 
            os.mkdir(os.path.abspath(fine_tune_meta_model_args.run_dir))

        fine_tune_meta_model_args.to_json_file(os.path.join(fine_tune_meta_model_args.run_dir, ARGS_FILENAME))


    return fine_tune_model(
        fine_tune_args, meta_model_args, sample_datum, train_dataloaders_by_data_frac,
        tqdm=tqdm, tuning_dataloader=val_dataloader, test_dataloader=test_dataloader,
    )