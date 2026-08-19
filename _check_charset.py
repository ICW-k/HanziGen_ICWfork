# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8")

tests = [
    (r"charsets\jf7000\jf7000_all.txt", "\u56fd", "国"),
    (r"charsets\jf7000\basic.txt", "\u56fd", "国"),
    (r"charsets\jf7000\jf7000_all.txt", "\u570b", "國"),
    (r"charsets\unihan\basic.txt", "\u56fd", "国"),
]

for path, ch, name in tests:
    try:
        with open(path, encoding="utf-8") as f:
            chars = {l.strip() for l in f if l.strip()}
        print(f"{path} | 字数: {len(chars)} | 含{name}(U+{ord(ch):04X}): {ch in chars}")
    except FileNotFoundError:
        print(f"{path} | 文件不存在")
