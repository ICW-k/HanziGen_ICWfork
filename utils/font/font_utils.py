from pathlib import Path

from tqdm.rich import tqdm


def get_font_paths(
    font_dir: str | Path,
) -> list[Path]:
    """
    Return all TTF and OTF file paths in the given directory.
    """
    font_dir = Path(font_dir)
    return sorted(
        [
            *font_dir.glob("*.ttf"),
            *font_dir.glob("*.otf"),
        ]
    )


# 参考字体优先级：环境变量配置，供 notebook / shell 覆盖，子进程会继承
REF_FONT_PRIORITY_ENV = "HANZIGEN_REF_FONT_PRIORITY"   # 逗号分隔的文件名，按优先级从高到低
REF_FONT_MODE_ENV = "HANZIGEN_REF_FONT_MODE"           # first=先命中优先（默认） / last=后写入覆盖（旧行为）
REF_FONT_STRICT_ENV = "HANZIGEN_REF_FONT_STRICT"       # 1/true=严格模式：只使用优先级列表中的字体


def resolve_reference_fonts(
    font_dir: str | Path,
) -> list[Path]:
    """
    返回按优先级排序的参考字体路径。

    优先级来源（从高到低）：
      1. 环境变量 HANZIGEN_REF_FONT_PRIORITY（逗号分隔的文件名，如 "jigmo.ttf,jigmo2.ttf"）
      2. 文件名排序（默认：jigmo.ttf < jigmo2.ttf < jigmo3.ttf）

    同名字符由多个参考字体同时覆盖时，调用方应配合 HANZIGEN_REF_FONT_MODE：
      - "first"（默认）：排在前的字体优先命中，后面的字体不再覆盖
      - "last"：旧行为，后写入的字体覆盖先写入的

    HANZIGEN_REF_FONT_STRICT=1 时启用严格模式：未列入优先级列表的字体完全不参与
    （风格彻底统一）。若过滤后没有任何字体可用，则回退为全量（避免推理因参考
    缺口直接报错），并打印警告。
    """
    import os

    paths = get_font_paths(font_dir)
    if not paths:
        return paths

    raw = os.environ.get(REF_FONT_PRIORITY_ENV, "")
    if not raw.strip():
        return paths

    wanted = [w.strip().lower() for w in raw.split(",") if w.strip()]
    ranked: list[Path] = []
    for name in wanted:
        for p in paths:
            if p.name.lower() == name and p not in ranked:
                ranked.append(p)

    strict = os.environ.get(REF_FONT_STRICT_ENV, "").strip().lower() in ("1", "true", "yes")
    if strict:
        if ranked:
            dropped = [p.name for p in paths if p not in ranked]
            if dropped:
                print(f"[参考字体] 严格模式：仅使用 {ranked[0].name} 等 {len(ranked)} 个指定字体，"
                      f"已排除 {dropped}")
            return ranked
        print("[WARN] 严格模式下优先级列表未命中任何字体，回退为使用全部参考字体")

    # 非严格：未列出的字体排在末尾作为兜底，保持原有相对顺序
    for p in paths:
        if p not in ranked:
            ranked.append(p)
    return ranked


def use_first_reference_font() -> bool:
    """
    是否采用"先命中的优先"语义（True）。False 表示旧行为"后写入覆盖"。
    """
    import os

    return os.environ.get(REF_FONT_MODE_ENV, "first").strip().lower() != "last"


def get_charset_paths(
    charset_dir: str | Path,
) -> list[Path]:
    """
    Return all charset file paths in the given directory.
    """
    charset_dir = Path(charset_dir)
    return sorted(
        [
            *charset_dir.glob("*.txt"),
        ]
    )


def read_charset_from_file(
    charset_path: str | Path,
) -> set[str]:
    """
    Read a charset file and return a set of characters.
    """
    charset_path = Path(charset_path)
    with charset_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file.readlines() if len(line.strip()) == 1}


def write_charset_to_file(
    charset: set[str],
    file_path: str | Path,
) -> None:
    """
    Write a charset to a file, one character per line.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        for char in sorted(charset):
            file.write(f"{char}\n")
