#!/usr/bin/env bash
# z2feat diagnostics, 6 arms, pinned to the FREE GPUs (0,1,4,5,6,7 -- skips busy 2,3).
# Two ideas x 3 rulers (random_drift, rff, random_linear):
#   (A) --whiten_h       : flatten the ruler spectrum -> does it fix z2feat's low-rank collapse?
#   (B) --width 16       : shrink the ruler to z's dim -> can 16-dim z now reconstruct H?
# Compare each against the raw z2feat (width 512, no whiten) arms already in runs/w512v2.
#
# Usage: DATA=data/imagenette2-160 EPOCHS=800 ./run_z2feat_diag.sh
set -uo pipefail
DATA="${DATA:-data/imagenette2-160}"; EPOCHS="${EPOCHS:-800}"; OUT="${OUT:-runs/z2diag}"
GPUS=(0 1 4 5 6 7)   # free GPUs (2,3 busy)

#            name                    reservoir       extra
rows=(
  "drift_whiten          random_drift   --direction z2feat --whiten_h --drift_alpha 0.1 --width 512"
  "rff_whiten            rff            --direction z2feat --whiten_h --width 512"
  "linear_whiten         random_linear  --direction z2feat --whiten_h --width 512"
  "drift_w16             random_drift   --direction z2feat --drift_alpha 0.1 --width 16"
  "rff_w16               rff            --direction z2feat --width 16"
  "linear_w16            random_linear  --direction z2feat --width 16"
)

i=0
for r in "${rows[@]}"; do
  set -- $r; name=$1; res=$2; shift 2; extra="$*"
  g=${GPUS[$i]}
  mkdir -p "$OUT/$name"
  echo "[gpu $g] $name: reservoir=$res $extra"
  CUDA_VISIBLE_DEVICES=$g python imagenette/train.py \
    --out "$OUT/$name" --data "$DATA" --method epi --lamb 0.1 \
    --reservoir "$res" $extra --proj_dim 16 --epochs "$EPOCHS" \
    > "$OUT/$name.log" 2>&1 &
  i=$((i + 1))
done
wait

echo "=== done. final / best acc + late erank per arm ==="
for r in "${rows[@]}"; do set -- $r; n=$1
  f="$OUT/$n/result.json"
  [ -f "$f" ] && python -c "import json;d=json.load(open('$f'));h=d['history'][-10:];import statistics as s;print(f\"  {'$n':20} final={d['final_acc']*100:5.2f} best={d['best_acc']*100:5.2f} erank={s.mean([e['erank'] for e in h]):5.1f}\")"
done
