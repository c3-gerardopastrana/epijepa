#!/usr/bin/env bash
# Width-512 concurrent run: the baselines + best label-free (drift) + random-linear ruler.
# All 800 epochs, one arm per GPU (no crashes), width=512 (the repo's Imagenette reservoir
# size -- fixes the earlier under-powered width=64).
#
# Usage: DATA=data/imagenette2-160 EPOCHS=800 ./run_w512.sh
set -uo pipefail
DATA="${DATA:-data/imagenette2-160}"; EPOCHS="${EPOCHS:-800}"; OUT="${OUT:-runs/w512}"; W=512

#            name          method  lamb  reservoir       extra
rows=(
  "sigreg          sigreg  0.02  random         "
  "epiplexity      epi     0.1   random         --direction feat2z"
  "drift           epi     0.1   random_drift   --direction feat2z --drift_alpha 0.1"
  "drift_z2feat    epi     0.1   random_drift   --direction z2feat --drift_alpha 0.1"
  "random_linear   epi     0.1   random_linear  --direction z2feat"
)

gpu=0
for r in "${rows[@]}"; do
  set -- $r; name=$1; method=$2; lamb=$3; res=$4; shift 4; extra="$*"
  mkdir -p "$OUT/$name"
  echo "[gpu $gpu] $name: method=$method lamb=$lamb reservoir=$res width=$W $extra"
  CUDA_VISIBLE_DEVICES=$gpu python imagenette/train.py \
    --out "$OUT/$name" --data "$DATA" --method "$method" --lamb "$lamb" \
    --reservoir "$res" $extra --proj_dim 16 --width "$W" --epochs "$EPOCHS" \
    > "$OUT/$name.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait

echo "=== done (width 512). FINAL / best acc per arm: ==="
for r in "${rows[@]}"; do set -- $r; n=$1
  f="$OUT/$n/result.json"
  [ -f "$f" ] && printf "  %-14s final=%s  best=%s\n" "$n" \
    "$(python -c "import json;print(round(json.load(open('$f'))['final_acc']*100,2))" 2>/dev/null)" \
    "$(python -c "import json;print(round(json.load(open('$f'))['best_acc']*100,2))" 2>/dev/null)"
done
