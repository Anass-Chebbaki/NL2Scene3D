# addon/fix_imports.py
"""
One-shot script: rewrite imports inside the add-on's core/ folder so
that the modules can be imported as `core.*` instead of `nl2scene3d.*`.

Run this once after copying core files from src/nl2scene3d/.
"""
from pathlib import Path
import re

CORE_DIR = Path(__file__).resolve().parent / "nl2scene3d_addon" / "core"

# Patterns to rewrite: nl2scene3d.X  ->  core.X
PATTERNS = [
    (re.compile(r"from\s+nl2scene3d\.(\w+)"), r"from core.\1"),
    (re.compile(r"import\s+nl2scene3d\.(\w+)"), r"import core.\1"),
    (re.compile(r"from\s+nl2scene3d\s+import"), r"from core import"),
]

count = 0
for py_file in CORE_DIR.glob("*.py"):
    text = py_file.read_text(encoding="utf-8")
    original = text
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    if text != original:
        py_file.write_text(text, encoding="utf-8")
        print(f"Fixed imports in: {py_file.name}")
        count += 1

print(f"\nDone. {count} files modified.")