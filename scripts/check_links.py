#!/usr/bin/env python3
"""Fail if any relative link or image in the Markdown files does not resolve, or if an in-page
anchor (#section) does not match a heading.

    python scripts/check_links.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|(?:src|href)=\"([^\"]+)\"")


def slug(h: str) -> str:
    h = re.sub(r"<[^>]+>", "", h).strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path: str) -> set[str]:
    out, seen = set(), {}
    in_code = False
    for line in open(path, encoding="utf-8"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("#"):
            s = slug(line.lstrip("#"))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def main() -> int:
    bad = 0
    files = [p for p in glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True)
             if os.sep + "." not in os.path.relpath(p, ROOT) or ".github" in p]
    for md in files:
        text = open(md, encoding="utf-8").read()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in LINK.finditer(text):
            target = m.group(1) or m.group(2)
            if re.match(r"^(https?:|mailto:)", target):
                continue
            path, _, frag = target.partition("#")
            dest = os.path.normpath(os.path.join(os.path.dirname(md), path)) if path else md
            if not os.path.exists(dest):
                print(f"{os.path.relpath(md, ROOT)}: missing {target}")
                bad += 1
            elif frag and dest.endswith(".md") and frag not in anchors(dest):
                print(f"{os.path.relpath(md, ROOT)}: no anchor #{frag} in {os.path.relpath(dest, ROOT)}")
                bad += 1
    print(f"checked {len(files)} markdown files: {'OK' if not bad else f'{bad} broken'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
