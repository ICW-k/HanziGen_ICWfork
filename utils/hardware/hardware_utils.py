import os

import torch
import torch.nn as nn


def select_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Select the optimal device for PyTorch operations.
    """
    if device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    if isinstance(device, torch.device):
        device = device
    else:
        device = torch.device(device)

    # 对固定输入尺寸启用 cuDNN benchmark：让 cuDNN 自动挑选最优卷积算法，
    # 显著加速 VQ-VAE / UNet 的前向与反向，且不改变任何数值结果。
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return device


def print_model_params(model: nn.Module) -> None:
    """
    Print the total number of parameters in a PyTorch model.
    """
    params = sum(p.numel() for p in model.parameters())

    if params >= 1e9:
        formatted_params = f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        formatted_params = f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        formatted_params = f"{params / 1e3:.2f}K"
    else:
        formatted_params = str(params)

    print(f"Total model parameters: {formatted_params}")


# ======================= 硬件自适应参数（auto 模式 + preset） =======================
#
# 背景：项目用 512x512 单通道字形 PNG 训练 VQ-VAE（像素空间，显存瓶颈），
# 再在 64x64 latent 空间训练 LDM（显存占用极小，瓶颈转 CPU 数据供给）。
# 因此无需为每个云实例维护专用脚本，运行时按真实硬件实时推算即可。
#
# 两种 preset（由 --preset 参数或脚本传入）：
#   - "aggressive"（云端，时间=金钱）：留 1 核给主进程，prefetch 拉满，
#     显存预留比例更高（0.95），尽可能压榨 GPU/CPU 吞吐。
#   - "conservative"（本地机）：留 2 核余量，prefetch 保守，显存预留更稳（0.85），
#     优先保证稳定、可后台挂机、不干扰日常使用。
#
# 推算依据（已用现有 4 档脚本实测值验算，见 scripts 内注释）：
#   - num_workers 由 CPU 核数决定：留出主进程/系统/预取余量，封顶 16。
#   - VQ-VAE batch 由显存决定：AMP 下双路(target+reference)前向+反传，
#     实测约 0.33 GB/样本，乘以安全系数后按显存反推。
#   - LDM batch 与显存基本无关：64x64 latent 极省显存，16G 即可支撑 128，
#     且 8 核数据供给已够喂满 128，故统一取 128（软上限，继续加大收益递减）。

# VQ-VAE 每样本显存（GB），AMP 开启下的双路前向+反传实测值。
_VQVAE_GB_PER_SAMPLE = 0.33

# LDM latent 空间 batch 软上限（8 核即可喂满，继续加大无收益）。
_LDM_MAX_BATCH = 128

# 各 preset 的调参表
_PRESETS = {
    "aggressive": {
        # 显存预留比例：aggressive 更高，吃满显存带宽
        "vram_reserve_fraction": 0.95,
        # 少核机器留 1 核给主进程（激进）
        "small_cores_reserve": 1,
        # 多核机器留的余量
        "large_cores_reserve": 3,
        # 预取倍数（prefetch_factor）
        "prefetch_factor": 8,
    },
    "conservative": {
        "vram_reserve_fraction": 0.85,
        "small_cores_reserve": 2,
        "large_cores_reserve": 6,
        "prefetch_factor": 4,
    },
}


def detect_hardware() -> dict:
    """
    检测当前设备的硬件信息，返回字典：
        gpu_available, gpu_name, vram_gb, cpu_cores, device_type
    """
    gpu_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    info = {
        "gpu_available": gpu_available,
        "mps_available": mps_available,
        "gpu_name": None,
        "vram_gb": 0.0,
        "cpu_cores": os.cpu_count() or 4,
        "device_type": "cuda" if gpu_available else ("mps" if mps_available else "cpu"),
    }
    if gpu_available:
        prop = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = prop.total_memory / 1024**3
    elif mps_available:
        info["gpu_name"] = "Apple Silicon GPU (MPS)"
    return info


def _preset_params(preset: str) -> dict:
    """返回指定 preset 的调参表，未知 preset 回退 conservative。"""
    return _PRESETS.get(preset, _PRESETS["conservative"])


def auto_num_workers(cpu_cores: int | None = None, preset: str = "conservative") -> int:
    """
    按 CPU 核数推算 DataLoader worker 数，留出主进程/系统/预取余量：
    - aggressive：少核留 1 核，多核留 3 核（压榨）
    - conservative：少核留 2 核，多核留 6 核（稳妥）
    封顶 16，至少 1。
    """
    cores = cpu_cores if cpu_cores is not None else (os.cpu_count() or 4)
    p = _preset_params(preset)
    if cores <= 8:
        return max(1, cores - p["small_cores_reserve"])
    return min(16, cores - p["large_cores_reserve"])


def auto_vqvae_batch_size(vram_gb: float, preset: str = "conservative") -> int:
    """
    按显存推算 VQ-VAE 训练 batch：显存 × 预留比例 ÷ 每样本显存，取整到 4 的倍数。
    无 GPU 时返回保守值 8（CPU 训练）。
    """
    if vram_gb <= 0:
        return 8
    p = _preset_params(preset)
    usable = vram_gb * p["vram_reserve_fraction"]
    batch = int(usable / _VQVAE_GB_PER_SAMPLE)
    batch = max(4, (batch // 4) * 4)  # 至少 4，且对齐 4 的倍数
    return batch


def auto_ldm_batch_size(vram_gb: float) -> int:
    """
    LDM 在 latent 空间训练，显存不紧张，统一取软上限 128；
    低显存（<20G，如 T4 16G）时取保守值 64，贴合原有 T4 档实测。
    """
    if vram_gb > 0 and vram_gb < 20:
        return 64
    return _LDM_MAX_BATCH


def auto_eval_batch_size(vram_gb: float) -> int:
    """
    评估批大小按显存分档：>=24G 取 16，否则取 8（保守）。
    """
    return 16 if vram_gb >= 24 else 8


def check_training_viability() -> dict:
    """
    检测本地机是否适合运行本项目训练，返回可读的判定结果。

    判定规则：
    - 有 NVIDIA CUDA GPU 且显存 >= 8GB → 可训练
    - 有 CUDA GPU 但显存 < 8GB → 显存不足，风险高
    - 仅 Apple Silicon (MPS) → 可尝试，但 VQ-VAE/LDM 部分算子兼容性需实测
    - 仅 CPU / AMD 核显等 → 无法训练（无可用加速器）

    返回 {"viable": bool, "reason": str, "device_type": str, ...}
    """
    info = detect_hardware()
    if info["gpu_available"]:
        if info["vram_gb"] >= 8:
            return {**info, "viable": True, "reason": "NVIDIA GPU 显存充足"}
        return {
            **info,
            "viable": False,
            "reason": f"NVIDIA GPU 显存仅 {info['vram_gb']:.1f}GB，低于训练最低建议 8GB",
        }
    if info["mps_available"]:
        return {
            **info,
            "viable": True,
            "reason": "Apple Silicon (MPS) 可用，可尝试训练（部分算子兼容性需实测）",
        }
    return {
        **info,
        "viable": False,
        "reason": "未检测到 NVIDIA CUDA GPU 或 Apple Silicon，仅 CPU/核显，无法运行本项目训练",
    }


def apply_auto_tuning(
    dataset_config,
    device: torch.device,
    mode: str,
    auto_batch: bool = True,
    auto_workers: bool = True,
    preset: str = "conservative",
) -> dict:
    """
    按硬件实时推算 batch_size / num_workers / prefetch，并（在调用方允许时）写回 config。

    device: 训练设备（仅用于判断是否 GPU）
    mode:   "vqvae" | "ldm"
    auto_batch:   True 时按显存/latent 空间推算 batch_size 并写回
    auto_workers: True 时按 CPU 核数推算 num_workers 并写回
    preset:       "aggressive"（云端压榨）| "conservative"（本地稳妥）

    返回实际采用的参数与硬件信息，便于打印。
    """
    info = detect_hardware()
    p = _preset_params(preset)
    workers = auto_num_workers(info["cpu_cores"], preset=preset)

    if mode == "vqvae":
        batch = auto_vqvae_batch_size(info["vram_gb"], preset=preset)
    else:  # ldm
        batch = auto_ldm_batch_size(info["vram_gb"])

    if auto_batch:
        dataset_config.batch_size = batch
    if auto_workers:
        dataset_config.num_workers = workers
    # prefetch_factor 若 config 支持则写入
    if hasattr(dataset_config, "prefetch_factor"):
        dataset_config.prefetch_factor = p["prefetch_factor"]

    return {
        **info,
        "preset": preset,
        "batch_size": dataset_config.batch_size,
        "num_workers": dataset_config.num_workers,
        "prefetch_factor": getattr(dataset_config, "prefetch_factor", p["prefetch_factor"]),
        "eval_batch_size": auto_eval_batch_size(info["vram_gb"]),
    }
