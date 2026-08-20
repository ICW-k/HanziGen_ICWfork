from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from configs.vqvae_config import VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from utils.hardware.hardware_utils import select_device

from .vqvae_encoder_decoder import VQVAEDecoder, VQVAEEncoder, VQVAEQuantizer


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE).
    """

    # ===== Initialization & Setup =====
    def __init__(
        self,
        model_config: VQVAEModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Initialize model device
        self.device = select_device(device)

        # Define encoder, vector quantizer, and decoder
        self.encoder = VQVAEEncoder(
            in_channels=model_config.input_img_channels,
            base_channels=model_config.encoder_base_channels,
            out_channels=model_config.latent_dim,
        )
        self.vector_quantizer = VQVAEQuantizer(
            codebook_size=model_config.codebook_size,
            latent_dim=model_config.latent_dim,
            commitment_cost=model_config.commitment_cost,
        )
        self.decoder = VQVAEDecoder(
            in_channels=model_config.latent_dim,
            base_channels=model_config.encoder_base_channels,
            out_channels=model_config.input_img_channels,
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Move model to device
        self.to(self.device)

    # ===== Core Operations =====
    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VQ-VAE model.
        """

        encoded_features = self.encoder(x)
        quantized_features, vq_loss, encoding_indices = self.vector_quantizer(
            encoded_features
        )
        x_recon = self.decoder(quantized_features)

        return x_recon, vq_loss, encoding_indices

    def _process_batch(
        self,
        batch: dict[str, Tensor],
        is_training: bool,
        optimizer: Optimizer | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> dict[str, float]:
        """
        Process a single batch of data.
        """
        tgt_imgs = batch["tgt_img"].to(self.device)
        ref_imgs = batch["ref_img"].to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            enabled=scaler.is_enabled() if scaler else False,
        ):
            tgt_x_recon, tgt_vq_loss, tgt_indices = self(tgt_imgs)
            tgt_recon_loss = self.loss_fn(tgt_x_recon, tgt_imgs)

            ref_x_recon, ref_vq_loss, ref_indices = self(ref_imgs)
            ref_recon_loss = self.loss_fn(ref_x_recon, ref_imgs)

            recon_loss = (tgt_recon_loss + ref_recon_loss) / 2
            vq_loss = (tgt_vq_loss + ref_vq_loss) / 2
            total_loss = recon_loss + vq_loss

        if is_training and optimizer is not None and scaler is not None:
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # ===== Compute Metrics =====
        all_indices = torch.cat([tgt_indices, ref_indices])
        perplexity = self._compute_perplexity(all_indices)
        codebook_usage = self._compute_codebook_usage(all_indices)

        metrics = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "vq": vq_loss.item(),
            "perplexity": perplexity.item(),
            "codebook_usage": codebook_usage,
        }

        return metrics

    # ===== Training and Validation =====
    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        epoch: int,
        min_val_loss: float,
    ) -> None:
        """
        Save a full training state (model + optimizer + scheduler + epoch).

        The dict holds a "model" key with the pure model state_dict so that inference /
        weight-loading code can read it compatibly.
        """
        ckpt = {
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "min_val_loss": min_val_loss,
        }
        if scaler is not None and scaler.is_enabled():
            ckpt["scaler"] = scaler.state_dict()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, checkpoint_path)

    def _restore_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
    ) -> tuple[int, float]:
        """
        Restore a full training state from a checkpoint. Returns (start_epoch, min_val_loss).

        Supports both the new full-state format (dict with "model" key) and the legacy
        pure state_dict format (weights-only resume, epoch restarts from 0).
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if isinstance(ckpt, dict) and "model" in ckpt:
            self.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and scaler.is_enabled() and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            return int(ckpt.get("epoch", 0)) + 1, float(
                ckpt.get("min_val_loss", float("inf"))
            )
        # 旧格式：纯 state_dict，只能恢复权重，epoch 从头开始
        self.load_state_dict(ckpt, strict=True)
        return 0, float("inf")

    def fit(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        training_config: VQVAETrainingConfig,
        resume_from: str | None = None,
    ) -> None:
        """
        Train the VQ-VAE model and save the best model checkpoint.

        When ``resume_from`` points to a full-state checkpoint, the optimizer / scheduler /
        epoch / best-loss are restored so training continues exactly where it stopped
        (true epoch-level resume).
        """
        log_dir = Path(training_config.tensorboard_log_dir) / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir) as writer:

            model_save_path = Path(training_config.model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

            start_epoch = 0
            min_val_loss = float("inf")

            if resume_from:
                if not Path(resume_from).exists():
                    raise FileNotFoundError(
                        f"[ERROR] No checkpoint to resume from: {resume_from}"
                    )
                start_epoch, min_val_loss = self._restore_checkpoint(
                    checkpoint_path=Path(resume_from),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )
                print(
                    f"[RESUME] 从 epoch {start_epoch} 继续训练（最佳 val loss: {min_val_loss:.6f}）"
                )

            ckpt_save_interval = getattr(training_config, "ckpt_save_interval", 0)
            val_every = max(1, getattr(training_config, "val_every", 5))

            for epoch in range(start_epoch, training_config.num_epochs):

                # 每 val_every 个 epoch 跑一次全量验证；最后一个 epoch 强制验证，
                # 保证最佳模型始终在完整评估后落盘
                do_validation = (epoch % val_every == 0) or (
                    epoch == training_config.num_epochs - 1
                )
                train_metrics, val_metrics = self._run_epoch(
                    loader=loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    do_validation=do_validation,
                )

                # Update learning rate
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # Log training and validation metrics
                self._log_training_metrics(
                    writer=writer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                )

                # Save the best model checkpoint (full training state)
                if val_metrics is not None and val_metrics["total"] < min_val_loss:
                    min_val_loss = val_metrics["total"]
                    self._save_checkpoint(
                        checkpoint_path=model_save_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        min_val_loss=min_val_loss,
                    )
                    print(f"✅ Best model saved (val loss: {min_val_loss:.6f})")
                elif ckpt_save_interval > 0 and epoch % ckpt_save_interval == 0:
                    # 周期保存完整状态，确保即使非最佳 epoch 也能精确续训
                    self._save_checkpoint(
                        checkpoint_path=model_save_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        min_val_loss=min_val_loss,
                    )
                    print(f"[CKPT] 周期性检查点已保存 (epoch {epoch})")

                # Print Metrics
                self._print_epoch_status(
                    epoch=epoch + 1,
                    total_epochs=training_config.num_epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                )

    def _run_epoch(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        do_validation: bool = True,
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        """
        Run one epoch of training, and validation when ``do_validation`` is set.
        """
        train_loader = loader.loader.train

        train_losses = self._train_one_epoch(train_loader, optimizer, scaler)
        if do_validation:
            val_loader = loader.loader.val
            val_losses = self._validate_one_epoch(val_loader)
        else:
            val_losses = None

        return train_losses, val_losses

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> dict[str, float]:
        """
        Train the model for one epoch.
        """
        self.train()
        epoch_metrics = {
            "total": 0.0,
            "recon": 0.0,
            "vq": 0.0,
            "perplexity": 0.0,
            "codebook_usage": 0.0,
        }

        for batch in tqdm(train_loader, desc="Training"):
            batch_metrics = self._process_batch(
                batch=batch,
                is_training=True,
                optimizer=optimizer,
                scaler=scaler,
            )
            for k in epoch_metrics.keys():
                epoch_metrics[k] += batch_metrics[k]

        num_batches = len(train_loader)
        return {
            metric_type: metric / num_batches
            for metric_type, metric in epoch_metrics.items()
        }

    @torch.no_grad()
    def _validate_one_epoch(
        self,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Validate the model for one epoch.
        """
        self.eval()
        epoch_metrics = {
            "total": 0.0,
            "recon": 0.0,
            "vq": 0.0,
            "perplexity": 0.0,
            "codebook_usage": 0.0,
        }

        for batch in tqdm(val_loader, desc="Validating"):
            batch_metrics = self._process_batch(
                batch=batch,
                is_training=False,
                optimizer=None,
            )
            for k in epoch_metrics.keys():
                epoch_metrics[k] += batch_metrics[k]

        num_batches = len(val_loader)
        return {
            metric_type: metric / num_batches
            for metric_type, metric in epoch_metrics.items()
        }

    # ===== Metrics =====
    def _compute_perplexity(self, indices: Tensor) -> Tensor:
        """
        Compute the perplexity of the codebook usage.
        """
        # Calculate the frequency of each index
        index_counts = torch.bincount(indices.view(-1), minlength=self.vector_quantizer.codebook_size)
        # Calculate the probability distribution
        index_probs = index_counts.float() / index_counts.sum()
        # Compute entropy, adding a small epsilon to avoid log(0)
        entropy = -torch.sum(index_probs * torch.log(index_probs + 1e-10))
        # Perplexity is the exponential of the entropy
        return torch.exp(entropy)

    def _compute_codebook_usage(self, indices: Tensor) -> float:
        """
        Compute the percentage of the codebook that is used.
        """
        # Count unique indices used
        unique_indices = torch.unique(indices)
        # Calculate the ratio of used indices to the total codebook size
        return (len(unique_indices) / self.vector_quantizer.codebook_size) * 100.0

    # ===== Logging =====
    def _log_training_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        learning_rate: float,
    ) -> None:
        """
        Log training and validation metrics to TensorBoard.
        """
        for metric_type, metric_value in train_metrics.items():
            writer.add_scalar(f"train/{metric_type}", metric_value, epoch)

        if val_metrics is not None:
            for metric_type, metric_value in val_metrics.items():
                writer.add_scalar(f"val/{metric_type}", metric_value, epoch)

        writer.add_scalar("LR", learning_rate, epoch)

    def _print_epoch_status(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: dict,
        val_metrics: dict,
        learning_rate: float,
    ) -> None:
        """
        Print the status of the current epoch using a rich table.
        """
        console = Console()
        table = Table(title=f"Epoch Status [{epoch}/{total_epochs}]")

        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Train", justify="right", style="magenta")
        table.add_column("Validation", justify="right", style="green")

        for metric_type in train_metrics.keys():
            val_value = (
                "-" if val_metrics is None else f"{val_metrics[metric_type]:.4f}"
            )
            table.add_row(
                metric_type.replace("_", " ").capitalize(),
                f"{train_metrics[metric_type]:.4f}",
                val_value,
            )

        table.add_row("Learning Rate", f"{learning_rate:.6f}", "-")

        console.print(table)
