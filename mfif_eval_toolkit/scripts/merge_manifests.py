#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("inputs", nargs="+", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
frames = [pd.read_csv(path) for path in a.inputs]
a.output.parent.mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True).to_csv(a.output, index=False)
print(f"[DONE] {a.output} rows={sum(len(x) for x in frames)}")
