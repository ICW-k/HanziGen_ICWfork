#!/bin/bash

TARGET_FONT_PATH="fonts/ChangguMingtiBold.otf"
REFERENCE_FONTS_DIR="fonts/jigmo/"
IMG_WIDTH=512
IMG_HEIGHT=512
SAMPLE_RATIO=1.0
NUM_WORKERS=4  # 字形渲染并行线程数（Cell 1 会按内存自适应调整；小内存实例请保持 ≤4，防止 OOM）


TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

SOURCE_CHARSET_PATH="charsets/unihan_coverage/${TARGET_FONT_NAME}/covered.txt"

python prepare_dataset.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --source_charset_path "$SOURCE_CHARSET_PATH" \
    --img_size "$IMG_WIDTH" "$IMG_HEIGHT" \
    --sample_ratio "$SAMPLE_RATIO" \
    --num_workers "$NUM_WORKERS"
