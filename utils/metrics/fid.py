import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid
from PIL import Image
from tqdm.rich import tqdm

from utils.hardware.hardware_utils import select_device
from utils.image.image_utils import get_image_paths


# ---- scipy 兼容层 ----
# scipy>=1.15 移除了 sqrtm 的 disp 参数，而 cleanfid 0.1.10 的 frechet_distance
# 仍以 sqrtm(..., disp=False) 调用并解包返回值。检测到新 scipy 时打补丁，
# 使旧调用方式保持可用（disp=False 时返回 (sqrtm, errest) 二元组，与旧版一致）。
import scipy.linalg as _linalg

_orig_sqrtm = _linalg.sqrtm


def _sqrtm_supports_disp() -> bool:
    try:
        _orig_sqrtm(np.eye(2, dtype=np.float64), disp=False)
        return True
    except TypeError:
        return False


if not _sqrtm_supports_disp():

    def _sqrtm_compat(A, disp=None):
        result = _orig_sqrtm(A)
        return (result, 0) if disp is False else result

    _linalg.sqrtm = _sqrtm_compat
    print("[兼容] 检测到 scipy>=1.15（sqrtm 无 disp 参数），已为 cleanfid 打补丁")


def compute_fid_score(
    gen_img_dir: str | Path,
    gt_img_dir: str | Path,
    batch_size: int,
    device: torch.device | None = None,
) -> float:
    """
    Computes the Fréchet Inception Distance (FID) score between generated and target images.
    """
    device = select_device(device)
    score = fid.compute_fid(
        fdir1=gen_img_dir,
        fdir2=gt_img_dir,
        batch_size=batch_size,
        device=device,
    )

    return score


def compute_fid_from_directories(
    gen_img_dir: str | Path,
    gt_img_dir: str | Path,
    batch_size: int,
    device: str | torch.device | None = None,
) -> float:
    """ """
    device = select_device(device)
    temp_dir = tempfile.mkdtemp()
    temp_gen_img_dir = os.path.join(temp_dir, "gen_rgb_img")
    temp_gt_img_dir = os.path.join(temp_dir, "gt_rgb_img")

    os.makedirs(temp_gen_img_dir, exist_ok=True)
    os.makedirs(temp_gt_img_dir, exist_ok=True)

    try:
        for img_path in tqdm(
            get_image_paths(gen_img_dir),
            desc="Converting images to RGB",
        ):
            img = Image.open(img_path).convert("L")
            rgb_img = Image.merge("RGB", (img, img, img))
            rgb_img.save(os.path.join(temp_gen_img_dir, img_path.name))

        for img_path in tqdm(
            get_image_paths(gt_img_dir),
            desc="Converting images to RGB",
        ):
            img = Image.open(img_path).convert("L")
            rgb_img = Image.merge("RGB", (img, img, img))
            rgb_img.save(os.path.join(temp_gt_img_dir, img_path.name))

        fid_score = compute_fid_score(
            gen_img_dir=temp_gen_img_dir,
            gt_img_dir=temp_gt_img_dir,
            batch_size=batch_size,
            device=device,
        )

    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} removed.")

    return fid_score
