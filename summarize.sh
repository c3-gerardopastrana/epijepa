#!/usr/bin/env bash
# Summarise runs: best acc, final acc, final effective-rank, final raw-S per run.
# Reads every result.json under the given dir(s) (default: runs/).
#
# Usage:  ./summarize.sh [runs_dir ...]
set -uo pipefail
DIRS=("${@:-runs}")

python - "${DIRS[@]}" <<'PY'
import json, sys, glob, os
rows = []
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "**", "result.json"), recursive=True):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        h = r.get("history", [])
        last = h[-1] if h else {}
        rows.append((
            round(r.get("best_acc", 0) * 100, 2),
            round(r.get("final_acc", 0) * 100, 2),
            round(last.get("erank", float("nan")), 1),
            round(last.get("raw_s", float("nan")), 2),
            r.get("reservoir"), r.get("direction"), r.get("multi_agg"),
            r.get("lamb"), r.get("seed"),
            os.path.relpath(f),
        ))
rows.sort(key=lambda x: -x[0])
hdr = ("best", "final", "erank", "rawS", "reservoir", "dir", "agg", "lamb", "seed", "path")
print(f"{hdr[0]:>6} {hdr[1]:>6} {hdr[2]:>6} {hdr[3]:>8}  {hdr[4]:<14} {hdr[5]:<7} {hdr[6]:<6} {hdr[7]:>4} {hdr[8]:>4}  {hdr[9]}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>6} {r[2]:>6} {r[3]:>8}  {str(r[4]):<14} {str(r[5]):<7} {str(r[6]):<6} {str(r[7]):>4} {str(r[8]):>4}  {r[9]}")
PY
