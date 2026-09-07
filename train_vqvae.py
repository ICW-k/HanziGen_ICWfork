import argparse

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import VQVAEDatasetConfig, VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from models import VQVAE
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import (
    apply_auto_tuning,
    print_model_params,
    probe_vqvae_batch_fits,
    select_device,
)


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Train VQVAE model")
    parser.add_argument(
        "--split_ratios", type=float, nargs=2, help="Train/val split ratios"
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size（不传则按显存自动推算）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of DataLoader workers（不传则按 CPU 核数自动推算）",
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--val_every",
        type=int,
        help="Validate once every N epochs (default 5; 1 = every epoch)",
    )
    parser.add_argument("--model_save_path", type=str, help="Model save path")
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")
    parser.add_argument(
        "--preset",
        type=str,
        default="conservative",
        choices=["aggressive", "conservative"],
        help="硬件自适应档位：aggressive=云端压榨（留 1 核+高预取）；"
        "conservative=本地稳妥（留 2 核+保守预取）",
    )
    parser.add_argument(
        "--vram_reserve_fraction",
        type=float,
        help="显存预留比例（可用于训练的那部分显存，0.1-1.0）。"
        "不传则取环境变量 HANZIGEN_VRAM_RESERVE_FRACTION 或 preset 默认值"
        "（aggressive 0.95 / conservative 0.85）。OOM 时调小，想提速则调大",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to VQVAE weights (.pth) to resume from (only loads model state)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    return parser.parse_args()


def train_vqvae(
    dataset_config: VQVAEDatasetConfig,
    model_config: VQVAEModelConfig,
    training_config: VQVAETrainingConfig,
    device: torch.device,
    resume_from: str | None = None,
):
    """ """
    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    vqvae = VQVAE(
        model_config=model_config,
        device=device,
    )

    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=training_config.learning_rate,
    )

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=training_config.num_epochs,
        eta_min=training_config.min_learning_rate,
    )

    scaler = GradScaler(enabled=training_config.mixed_precision)

    print_model_params(
        model=vqvae,
    )

    # resume_from 由 fit 内部恢复完整状态（model+optimizer+scheduler+epoch）
    vqvae.fit(
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        training_config=training_config,
        resume_from=resume_from,
    )


def main() -> None:
    """ """
    args = parse_args()
    dataset_config = update_config_from_args(
        converting_config=VQVAEDatasetConfig(),
        args=args,
    )
    model_config = update_config_from_args(
        converting_config=VQVAEModelConfig(),
        args=args,
    )
    training_config = update_config_from_args(
        converting_config=VQVAETrainingConfig(),
        args=args,
    )
    device = select_device(args.device)

    # 硬件自适应：batch_size / num_workers 未显式传入时按真实硬件实时推算，
    # 消除"多配置专用脚本"，并保证 GPU 与 CPU 数据供给同步（避免线程饥饿 / GPU 空转）。
    auto_batch = args.batch_size is None
    tuning = apply_auto_tuning(
        dataset_config,
        device,
        mode="vqvae",
        auto_batch=auto_batch,
        auto_workers=args.num_workers is None,
        preset=args.preset,
        vram_reserve_fraction=args.vram_reserve_fraction,
    )
    if tuning["gpu_available"]:
        print(
            f"[硬件] GPU: {tuning['gpu_name']} ({tuning['vram_gb']:.1f} GB) | "
            f"CPU 核数: {tuning['cpu_cores']} | 档位: {tuning['preset']} | "
            f"显存预留: {tuning['vram_reserve_fraction']:.2f}"
        )
    else:
        print(
            f"[硬件] 未检测到 CUDA GPU（{tuning['cpu_cores']} 核） | "
            f"档位: {tuning['preset']}"
        )
    print(
        f"[自适应] batch_size={tuning['batch_size']} | "
        f"num_workers={tuning['num_workers']} | "
        f"prefetch_factor={tuning['prefetch_factor']}"
    )

    # 显存实测：auto 推算的 batch 依赖经验值（0.33GB/样本），训练开始前用 dry-run
    # 验证一次前向+反传，OOM 则自动减半降档；显式指定的 batch 只提示不改动
    if tuning["gpu_available"]:
        probe_model = VQVAE(model_config=model_config, device=device)
        probed = probe_vqvae_batch_fits(
            model=probe_model,
            batch_size=dataset_config.batch_size,
            mixed_precision=training_config.mixed_precision,
        )
        del probe_model
        torch.cuda.empty_cache()
        if probed < dataset_config.batch_size:
            if auto_batch:
                print(
                    f"[显存实测] batch {dataset_config.batch_size} 超出实际可用显存，"
                    f"自动降为 {probed}"
                )
                dataset_config.batch_size = probed
            else:
                print(
                    f"[WARN] 显存实测：batch {dataset_config.batch_size} 可能 OOM"
                    f"（实测上限约 {probed}），仍按指定值继续，OOM 时请手动调小"
                )

    train_vqvae(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        device=device,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
