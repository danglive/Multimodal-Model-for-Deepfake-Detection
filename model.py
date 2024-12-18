import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchmetrics
import albumentations as A
from utils import calc_prob
from pytorch_lightning.core.lightning import LightningModule
import wandb
from mmaction.models import TimeSformer

# Initialize metrics
train_auc = torchmetrics.AUROC(pos_label=1)
val_auc = torchmetrics.AUROC(pos_label=1)
test_auc = torchmetrics.AUROC(pos_label=1)

class Model(LightningModule):
    """
    PyTorch Lightning module for deepfake detection using a multimodal model:
    - Uses TimeSformer for audiovisual data.
    - A small latency network for SyncNet features.
    - A linear head for final classification.

    Training steps:
    - Compute loss with cross entropy and weighted samples.
    - Track AUC using torchmetrics.
    - Use ReduceLROnPlateau for LR scheduling based on validation AUC.
    """
    def __init__(self, total_steps, lr):
        super().__init__()
        # Initialize TimeSformer from mmaction2
        self.model = TimeSformer(
            num_frames=32,
            img_size=224,
            patch_size=16,
            embed_dims=768,
            attention_type='divided_space_time'
        )

        # Latency network to process SyncNet features (33-dim)
        self.latency_network = nn.Sequential(
            nn.Linear(33, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(768 + 8, 2),
        )

        self.best_auc = 0
        self.total_steps = total_steps
        self.lr = lr
        self.automatic_optimization = False

    def log_all(self, metrics):
        """Log metrics to wandb at each global step."""
        wandb.log(metrics, step=self.global_step)

    def forward(self, batch):
        # Extract video and latency input from batch
        audio_video = self.model(batch["video"])
        latency = self.latency_network(batch["latency"])
        concat_input = torch.cat([audio_video, latency], dim=1)
        logits = self.head(concat_input)
        return logits

    def training_step(self, batch, _):
        opt = self.optimizers()
        opt.zero_grad()
        lr = [group['lr'] for group in opt.param_groups][0]

        logits = self.forward(batch)
        loss = F.cross_entropy(logits, batch["label"], reduction="none")
        loss = (batch["weight"] * loss).sum() / batch["weight"].sum()

        mean_loss = torch.mean(self.all_gather(loss))
        train_auc.update(calc_prob(logits), batch["label"])

        self.manual_backward(loss)
        opt.step()

        if self.global_rank == 0:
            self.log_all({"train/loss": mean_loss, "lr": lr})

    def training_epoch_end(self, _):
        auc = train_auc.compute()
        train_auc.reset()

        if self.global_rank == 0:
            self.log_all({"train/auc": auc})

    def validation_step(self, batch, _):
        logits = self.forward(batch)
        loss = F.cross_entropy(logits, batch["label"], reduction="none")
        loss = (batch["weight"] * loss).sum() / batch["weight"].sum()

        mean_loss = torch.mean(self.all_gather(loss))
        val_auc.update(calc_prob(logits), batch["label"])

        if self.global_rank == 0:
            self.log_all({"val/loss": mean_loss})

    def validation_epoch_end(self, _):
        auc = val_auc.compute()
        val_auc.reset()

        sch = self.lr_schedulers()
        sch.step(auc)

        if self.global_rank == 0:
            self.best_auc = max(auc, self.best_auc)
            self.log_all({
                "val/auc": auc,
                "best_auc": self.best_auc,
                "epoch": self.current_epoch
            })
        self.log("val/auc", auc, sync_dist=True)

    def test_step(self, batch, _):
        logits = self.forward(batch)
        test_auc.update(calc_prob(logits), batch["label"])

    def test_epoch_end(self, _):
        auc = test_auc.compute()
        test_auc.reset()
        if self.global_rank == 0:
            self.log_all({"test/auc": auc})
        self.log("test/auc", auc, sync_dist=True)

    def predict_step(self, batch, _):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max",
            patience=3,
            cooldown=1,
            factor=0.3,
            verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/auc"}