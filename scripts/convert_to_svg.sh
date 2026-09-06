#!/bin/bash

TARGET_FONT_PATH="fonts/myfont.ttf"
BLACKLEVEL=0.5
TURDSIZE=2
ALPHAMAX=1.0
OPTTOLERANCE=0.2
NUM_WORKERS=auto                       # 并行转换进程数：auto=按 CPU 核数自动（核数-2、封顶 32）
                                       #   手动指定：填入整数；填 1 = 串行


TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

INPUT_DIR="samples_${TARGET_FONT_NAME}/inference/gen"
OUTPUT_DIR="svgs_${TARGET_FONT_NAME}"

ARGS=()
if [ "$NUM_WORKERS" != "auto" ]; then
  ARGS+=(--num_workers "$NUM_WORKERS")
fi

python convert_to_svg.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --blacklevel "$BLACKLEVEL" \
    --turdsize "$TURDSIZE" \
    --alphamax "$ALPHAMAX" \
    --opttolerance "$OPTTOLERANCE" \
    "${ARGS[@]}"
