#!/usr/bin/env bash
# Width-512, 800-epoch comparison, one arm per GPU. Chassis is set PER DIRECTION:
#   feat2z arms -> --no_embed_bn (the faithful Imagenette chassis; BN hurt feat2z epi)
#   z2feat arms -> BN on (default; z2feat needs it for embedding-scale control)
# sigreg auto-disables BN regardless. This fixes the earlier run where BN was on for all.
# NOTE: the *_nobn z2feat arms test whether z2feat needs the BN at all (lambda=0.1 may let
# the invariance term restrain scale). They MAY blow up (watch inv -> if it climbs past ~2
# and keeps rising, that arm is diverging). Divergence is an informative result, not a bug.
#
# Usage: DATA=data/imagenette2-160 EPOCHS=800 ./run_w512.sh   (needs 8 GPUs)
set -uo pipefail
DATA="${DATA:-data/imagenette2-160}"; EPOCHS="${EPOCHS:-800}"; OUT="${OUT:-runs/w512v2}"; W=512

#            name            method  lamb  reservoir       extra (incl. direction + BN)
rows=(
  "sigreg          sigreg  0.02  random         "
  "epiplexity      epi     0.1   random         --direction feat2z --no_embed_bn"
  "drift           epi     0.1   random_drift   --direction feat2z --no_embed_bn --drift_alpha 0.1"
  "drift_z2feat    epi     0.1   random_drift   --direction z2feat --drift_alpha 0.1"
  "random_linear   epi     0.1   random_linear  --direction z2feat"
  "rff             epi     0.1   rff             --direction z2feat"
  "random_linear_nobn  epi 0.1   random_linear  --direction z2feat --no_embed_bn"
  "rff_nobn        epi     0.1   rff             --direction z2feat --no_embed_bn"
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
