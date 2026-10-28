# -*- coding: utf-8 -*-
"""Point configs/cucumber_dataset.yaml at the absolute location of this archive."""
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <archive>/code
ROOT = HERE.parent                              # <archive>
YAML = HERE / "configs" / "cucumber_dataset.yaml"

out, done = [], False
for line in YAML.read_text(encoding="utf-8").splitlines():
    if line.startswith("path:") and not line.lstrip().startswith("path: data"):
        out.append("path: %s" % ROOT.as_posix())
        done = True
    else:
        out.append(line)
YAML.write_text("\n".join(out) + "\n", encoding="utf-8")
print(("dataset root set to " if done else "no path: line found; please edit by hand: ") + str(ROOT))
