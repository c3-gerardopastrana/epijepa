#!/usr/bin/env bash
# Width-512, 800-epoch comparison, one arm per GPU (8 arms = 8 GPUs). BN set PER DIRECTION:
#   feat2z arms -> --no_embed_bn (faithful chassis; BN whitens the ridge TARGET z and hurts)
#   z2feat arms -> BN on (default; pins the ridge REGRESSOR z's scale, prevents blow-up)
# sigreg auto-disables BN. Each of drift / rff / random_linear is run BOTH directions.
#
# Usage: DATA=data/imagenette2-160 EPOCHS=800 ./run_w512.sh   (needs 8 GPUs)
set -uo pipefail
DATA="${DATA:-data/imagenette2-160}"; EPOCHS="${EPOCHS:-800}"; OUT="${OUT:-runs/w512v2}"; W=512

#            name                  method  lamb  reservoir       extra (direction + BN)
rows=(
  "sigreg                sigreg  0.02  random         "
  "epiplexity            epi     0.1   random         --direction feat2z --no_embed_bn"
  "drift_feat2z          epi     0.1   random_drift   --direction feat2z --no_embed_bn --drift_alpha 0.1"
  "drift_z2feat          epi     0.1   random_drift   --direction z2feat --drift_alpha 0.1"
  "rff_feat2z            epi     0.1   rff            --direction feat2z --no_embed_bn"
  "rff_z2feat            epi     0.1   rff            --direction z2feat"
  "random_linear_feat2z  epi     0.1   random_linear  --direction feat2z --no_embed_bn"
  "random_linear_z2feat  epi     0.1   random_linear  --direction z2feat"
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
