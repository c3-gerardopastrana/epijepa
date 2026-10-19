#!/usr/bin/env bash
# The 8 headline experiments at full budget (800 epochs), one per GPU, one wave.
#
# Picks: the published baselines (SIGReg, EpiJEPA, pretrained) + the strong label-free
# simples (z2feat, drift) + the min-max thesis (dyn-max iid, dyn-max multiscale, and its
# mean control). Each contender has its matched control; each method uses its own correct
# head (SIGReg raw via auto-disable, others BN) and its own tuned lambda.
#
# Report FINAL acc (not best) for fairness. ~6-8h/arm; 8 GPUs => ~one wave.
#
# Usage: DATA=data/imagenette2-160 EPOCHS=800 ./run_final8.sh
set -uo pipefail
DATA="${DATA:-data/imagenette2-160}"; EPOCHS="${EPOCHS:-800}"; OUT="${OUT:-runs/final8}"

#            name            method  lamb  reservoir           agg    extra
rows=(
  "sigreg          sigreg  0.02  random              stack  "
  "epijepa_orig    epi     0.1   random              stack  --direction feat2z"
  "pretrained      epi     0.1   pretrained          stack  --direction feat2z"
  "z2feat          epi     0.1   random              stack  --direction z2feat"
  "drift           epi     0.1   random_drift        stack  --direction feat2z --drift_alpha 0.1"
  "dynmax_k8       epi     0.1   minmax_random       max    --direction z2feat --K 8 --softmin_tau 1.0"
  "ms_dynmax_k8    epi     0.1   multiscale_minmax   max    --direction z2feat --K 8 --softmin_tau 1.0"
  "ms_mean_k8      epi     0.1   multiscale_minmax   mean   --direction z2feat --K 8"
)

gpu=0
for r in "${rows[@]}"; do
  set -- $r; name=$1; method=$2; lamb=$3; res=$4; agg=$5; shift 5; extra="$*"
  mkdir -p "$OUT/$name"
  echo "[gpu $gpu] $name: method=$method lamb=$lamb reservoir=$res agg=$agg $extra"
  CUDA_VISIBLE_DEVICES=$gpu python imagenette/train.py \
    --out "$OUT/$name" --data "$DATA" --method "$method" --lamb "$lamb" \
    --reservoir "$res" --multi_agg "$agg" $extra \
    --proj_dim 16 --width 64 --epochs "$EPOCHS" \
    > "$OUT/$name.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait

echo "=== done. FINAL acc per arm (fair metric): ==="
for r in "${rows[@]}"; do set -- $r; n=$1
  f="$OUT/$n/result.json"
  [ -f "$f" ] && printf "  %-14s final=%s  best=%s\n" "$n" \
    "$(python -c "import json;d=json.load(open('$f'));print(round(d['final_acc']*100,2))" 2>/dev/null)" \
    "$(python -c "import json;d=json.load(open('$f'));print(round(d['best_acc']*100,2))" 2>/dev/null)"
done
