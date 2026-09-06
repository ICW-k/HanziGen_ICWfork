import argparse

from configs import ConvertingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.image import GlyphImageConverter


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Convert images to SVG format")
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--blacklevel", type=float, help="Black level")
    parser.add_argument(
        "--turdsize", type=int, help="Suppress speckles of up to this size"
    )
    parser.add_argument("--alphamax", type=float, help="Corner threshold parameter")
    parser.add_argument(
        "--opttolerance", type=float, help="Curve optimization tolerance"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="并行转换进程数（不传则按 CPU 核数自动推算：核数-2、封顶 32）",
    )

    return parser.parse_args()


def convert_to_svg(
    input_dir: str,
    output_dir: str,
    converting_config: ConvertingConfig,
    num_workers: int | None = None,
) -> None:
    """ """
    converter = GlyphImageConverter(
        converting_config=converting_config,
    )
    converter.convert_images_to_svgs(
        input_dir=input_dir,
        output_dir=output_dir,
        num_workers=num_workers,
    )


def main() -> None:
    """ """
    args = parse_args()
    converting_config = update_config_from_args(
        converting_config=ConvertingConfig(),
        args=args,
    )

    convert_to_svg(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        converting_config=converting_config,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
