import pytorch_lightning
import torch
import time
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from utils import log, set_seed, predict
import numpy as np
import wandb
from data_loader import DataLoader
from model import Model
import json

def main(args):
    # Load config
    config = json.load(open("config.json"))
    set_seed(config["seed"])
    num_gpu = 0 if config["device"] == "cpu" else torch.cuda.device_count()

    # Initialize WandB
    wandb_config = {
        "data_version":     args.data_version,
        "code_version":     args.code_version,
        "pret_version":     args.pret_version,
        "num_gpu":          num_gpu,
        "batch_size":       config["train_batch_size"] * num_gpu,
        "image_size":       config["frame_size"],
        "max_lr":           config["lr"]
    }
    run = wandb.init(project="deepfake", config=wandb_config)

    # Load data
    data_module = DataLoader(
        args.data_version,
        config["root_path"],
        config["train_batch_size"]
    )

    # Calculate total steps (for scheduling, if needed)
    total_steps = config["epoch"] * int(np.ceil(data_module.get_trainset_size() / config["train_batch_size"]))
    model = Model(total_steps=total_steps, lr=config['lr'])

    # Load pre-trained weights if provided
    pretrain = torch.load(f"{config['root_path']}/pretrain/{args.pret_version}/model.ckpt")
    if "state_dict" in pretrain:
        pretrain = pretrain["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(pretrain, strict=False)
    log("missing_keys   :\t", missing_keys)
    log("unexpected_keys:\t", unexpected_keys)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/auc",
        dirpath=f"{config['root_path']}/output/{run.id}",
        filename="model",
        save_top_k=1,
        mode="max",
    )

    # Trainer setup
    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        max_epochs=config["epoch"],
        gpus=num_gpu,
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=1,
        precision=32 if config["device"] == "cpu" else 16,
    )

    # Train
    trainer.fit(model, data_module)

    # Test using the best checkpoint
    best_path = f"{config['root_path']}/output/{run.id}/model.ckpt"
    state_dict = torch.load(best_path)["state_dict"]
    model.load_state_dict(state_dict)

    trainer.test(model, [data_module.test_dataloader()])

    # Generate predictions for validation and test sets
    val_preds  = pd.DataFrame(predict(trainer, model, data_module.val_dataloader()))
    test_preds = pd.DataFrame(predict(trainer, model, data_module.test_dataloader()))

    val_preds.to_csv("val_preds.txt", index=False)
    test_preds.to_csv("test_preds.txt", index=False)

    wandb.save("val_preds.txt")
    wandb.save("test_preds.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the deepfake detection model.')
    parser.add_argument('--data-version', type=str, required=True, help="Data version identifier")
    parser.add_argument('--code-version', type=str, required=True, help="Code version identifier")
    parser.add_argument('--pret-version', type=str, required=True, help="Pretrained version identifier")
    args = parser.parse_args()
    main(args)