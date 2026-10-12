#!/usr/bin/env bash
# Diverse-RFF aggregation sweep, parallelized across GPUs (one arm per GPU).
#
# Two diversity modes via KERNEL_FAMILY (run on two branches, or two invocations):
#   rbf   : K bandwidths of one RBF kernel      (same kernel, different scales)
#   mixed : K different kernel families         (RBF/Laplacian/Cauchy/Matern)
#
# 8 arms total = {base, concat, mean, max} x {lambda 1.0, 0.1}. With 6+ GPUs these run
# concurrently (base_l1 is the single-kernel 'rff' reference; the rest are multi_rff).
#
# Usage:
#   DATA=data/imagenette2-160 EPOCHS=120 KERNEL_FAMILY=rbf   NGPU=6 ./run_rff.sh
#   DATA=data/imagenette2-160 EPOCHS=120 KERNEL_FAMILY=mixed NGPU=6 ./run_rff.sh
set -uo pipefail

DATA="${DATA:-data/imagenette2-160}"
EPOCHS="${EPOCHS:-120}"
KF="${KERNEL_FAMILY:-rbf}"
NGPU="${NGPU:-6}"
OUT="${OUT:-runs/rff_$KF}"
K="${K:-4}"
TAU="${TAU:-0.5}"

# arm spec: name method-args...  (reservoir/agg per arm; base uses single-kernel rff)
declare -a NAMES=(base_l1   concat_l1 mean_l1   max_l1    base_l01  concat_l01 mean_l01  max_l01)
declare -a LAMB=( 1.0       1.0       1.0       1.0       0.1       0.1        0.1       0.1)
declare -a RES=(  rff       multi_rff multi_rff multi_rff rff       multi_rff  multi_rff multi_rff)
declare -a AGG=(  stack     stack     mean      max       stack     stack      mean      max)

pids=()
for i in "${!NAMES[@]}"; do
  gpu=$(( i % NGPU ))
  name="${NAMES[$i]}"; lamb="${LAMB[$i]}"; res="${RES[$i]}"; agg="${AGG[$i]}"
  mkdir -p "$OUT/$name"
  echo "[gpu $gpu] $name: reservoir=$res agg=$agg lamb=$lamb kernel_family=$KF"
  CUDA_VISIBLE_DEVICES=$gpu python imagenette/train.py \
    --out "$OUT/$name" --data "$DATA" --method epi --lamb "$lamb" \
    --reservoir "$res" --multi_agg "$agg" --K "$K" --softmin_tau "$TAU" \
    --kernel_family "$KF" --proj_dim 16 --width 64 --epochs "$EPOCHS" \
    > "$OUT/$name.log" 2>&1 &
  pids+=($!)
  # throttle: never launch more than NGPU at once
  if (( (i + 1) % NGPU == 0 )); then wait "${pids[@]}"; pids=(); fi
done
wait "${pids[@]}" 2>/dev/null

echo "=== done ($KF). best acc per arm: ==="
for n in "${NAMES[@]}"; do
  f="$OUT/$n/result.json"
  [ -f "$f" ] && printf "  %-12s %s\n" "$n" "$(python -c "import json;print(round(json.load(open('$f'))['best_acc']*100,2))" 2>/dev/null)"
done
