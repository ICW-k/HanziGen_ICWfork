import argparse
import shutil
from pathlib import Path

from configs import FontProcessingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.font.font_utils import read_charset_from_file
from utils.image import GlyphImageGenerator


def _covered_chars(font_name: str, coverage_dir: str) -> set[str]:
    """读取某参考字体在 unihan 覆盖率分析下已覆盖的字符集。"""
    p = Path(coverage_dir) / font_name / "covered.txt"
    if not p.exists():
        return set()
    try:
        return read_charset_from_file(p)
    except Exception:
        return set()


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Prepare image dataset for training")
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument(
        "--reference_fonts_dir", type=str, help="Reference fonts directory"
    )
    parser.add_argument("--source_charset_path", type=str, help="Source charset path")
    parser.add_argument(
        "--img_size", type=int, nargs=2, help="Image size (width height)"
    )
    parser.add_argument("--sample_ratio", type=float, help="Sampling ratio (0-1)")
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker threads for glyph rendering (match CPU cores)",
    )

    return parser.parse_args()


def prepare_image_dataset(
    target_font_path: str,
    reference_fonts_dir: str,
    source_charset_path: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    """ """
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    tgt_generator = GlyphImageGenerator.from_target_font(
        target_font_path=target_font_path,
        font_processing_config=font_processing_config,
    )
    ref_generators = GlyphImageGenerator.from_reference_fonts(
        reference_fonts_dir=reference_fonts_dir,
        font_processing_config=font_processing_config,
    )
    # 按优先级排序（默认文件名顺序：jigmo → jigmo2 → jigmo3）
    try:
        from utils.font.font_utils import resolve_reference_fonts, use_first_reference_font

        ranked = resolve_reference_fonts(reference_fonts_dir)
        order = {p.name: i for i, p in enumerate(ranked)}
        ref_generators = sorted(
            ref_generators, key=lambda g: order.get(Path(g.font_path).name, 999)
        )
    except Exception:
        use_first_reference_font = None

    if not ref_generators:
        # 目录不存在或为空时 glob 返回空列表且不报错，会导致 data/reference 静默缺失，
        # 直到 extract_charset 阶段才抛出难以定位的错误，故在此提前失败并给出明确指引。
        raise FileNotFoundError(
            f"参考字体目录 '{reference_fonts_dir}' 下没有找到任何 .ttf / .otf 文件，"
            f"data/reference 将无法生成。\n"
            f"请确认 Jigmo 参考字体（jigmo.ttf / jigmo2.ttf / jigmo3.ttf）已放入该目录；"
            f"若缺失，运行 notebook 的 Cell 1 会自动准备，或手动从 "
            f"https://kamichikoichi.github.io/jigmo/ 下载后解压到该目录。"
        )

    tgt_generator.generate_glyph_images(
        source_charset_path=source_charset_path,
        font_role="target",
    )

    # 参考字形：同一字符常被多个参考字体同时覆盖。默认采用"先命中的优先"
    # （优先级高的字体先写，后面的字体跳过已生成的字），避免生僻字体覆盖常用字形；
    # 设 HANZIGEN_REF_FONT_MODE=last 可恢复旧行为（后写入覆盖）。
    first_wins = True
    try:
        from utils.font.font_utils import use_first_reference_font as _first

        first_wins = _first()
    except Exception:
        pass

    saved: set[str] = set()
    for ref_generator in ref_generators:
        ref_generator.generate_glyph_images(
            source_charset_path=source_charset_path,
            font_role="reference",
            exclude_chars=saved if first_wins else None,
        )
        if first_wins:
            saved |= _covered_chars(
                ref_generator.font_name,
                font_processing_config.unihan_coverage_charset_dir,
            )


def main() -> None:
    """ """
    args = parse_args()
    font_processing_config = update_config_from_args(
        converting_config=FontProcessingConfig(),
        args=args,
    )

    prepare_image_dataset(
        target_font_path=args.target_font_path,
        reference_fonts_dir=args.reference_fonts_dir,
        source_charset_path=args.source_charset_path,
        font_processing_config=font_processing_config,
    )


if __name__ == "__main__":
    main()
