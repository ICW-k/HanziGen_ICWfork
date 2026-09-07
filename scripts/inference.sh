#!/bin/bash

TARGET_FONT_PATH="fonts/myfont.ttf"
REFERENCE_FONTS_DIR="fonts/jigmo/"
BATCH_SIZE=64                         # 推理批大小：LDM 在 latent 空间（64x64）采样，显存占用极低，
                                      #   调大可提高 GPU 利用率且不损失精度（每个字独立采样）。
                                      #   16G 显存可设 64-128；OOM 则减半重试
NUM_WORKERS=auto                      # 推理 DataLoader 并行进程数：auto=按 CPU 核数 auto 推算（留 2 核、封顶 8）；
                                      #   PNG 读盘是 IO 密集，并行可显著提速；手动指定填整数（如 8）
SAMPLE_STEPS=50                        # 去噪采样步数（DDIM）：从随机噪声逐步去噪生成字形的迭代次数
                                       #   步数越多 → 细节越准、字形错误越少，但推理时间线性变长
                                       #   50=默认折中；结果不好可试 100（质量更好、耗时翻倍）；
                                       #   只想快看效果可试 20（约 2.5 倍速，质量明显下降）
IMG_WIDTH=512
IMG_HEIGHT=512
DEVICE="cuda"


TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

CHARSET_PATH="charsets/jf7000_coverage/${TARGET_FONT_NAME}/missing.txt"
PRETRAINED_LDM_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

# NUM_WORKERS 为 auto 时按 CPU 核数 auto 推算（不传参，由 inference.py 推算）；仅显式数字时传入覆盖
ARGS=()
if [ "$NUM_WORKERS" != "auto" ]; then
  ARGS+=(--num_workers "$NUM_WORKERS")
fi

python inference.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --charset_path "$CHARSET_PATH" \
    --pretrained_ldm_path "$PRETRAINED_LDM_PATH" \
    --batch_size "$BATCH_SIZE" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_size "$IMG_WIDTH" "$IMG_HEIGHT" \
    --device "$DEVICE" \
    "${ARGS[@]}"