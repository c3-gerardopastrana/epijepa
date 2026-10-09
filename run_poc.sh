#!/usr/bin/env bash
# PoC #1: does a stronger *fixed* ruler (frozen pretrained ResNet) beat the
# frozen *random* reservoir as the epiplexity anti-collapse signal?
#
# Runs four arms at a matched (shrunk) epoch budget so the random-vs-pretrained
# gap is a fair read. The key comparison is the epiplexity-only pair (view
# agreement off, lamb=1.0): if pretrained lifts test_acc over random there,
# ruler quality was the bottleneck and PoC #2 (reconstruction ruler) is worth it.
#
# Usage:  DATA=data/imagenette2-160 EPOCHS=60 ./run_poc.sh
set -euo pipefail

DATA="${DATA:-data/imagenette2-160}"
EPOCHS="${EPOCHS:-60}"
OUT="${OUT:-runs/imagenette}"
LAMB="${LAMB:-0.1}"           # EpiJEPA (score + view agreement)
LAMB_ONLY="${LAMB_ONLY:-1.0}" # epiplexity-only (view agreement off)

run () {  # name  method  lamb  reservoir
  echo "=== $1 (method=$2 lamb=$3 reservoir=$4 epochs=$EPOCHS) ==="
  python imagenette/train.py --out "$OUT/$1" --data "$DATA" \
    --method "$2" --lamb "$3" --reservoir "$4" --epochs "$EPOCHS"
}

# EpiJEPA: score + view agreement
run epi_random      epi "$LAMB"      random
run epi_pretrained  epi "$LAMB"      pretrained

# epiplexity-only: the go/no-go signal (view agreement off)
run epi_only_random     epi "$LAMB_ONLY" random
run epi_only_pretrained epi "$LAMB_ONLY" pretrained

echo "=== done. compare best test_acc in $OUT/*/result.json ==="
echo "key read: epi_only_pretrained vs epi_only_random (published random epi-only ~37.35%)"
