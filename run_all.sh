#!/usr/bin/env bash
# Runs every experiment behind the README tables. Safe to re-run: finished steps are skipped.
#   bash run_all.sh 2>&1 | tee run_all.log
set -e
AUG="data/aug_1 data/aug_2 data/aug_3 data/aug_4"
SCR="data/scripted_1 data/scripted_2 data/scripted_3 data/scripted_4"

echo "== 1/4 baseline datasets"
[ -d data/raw ] || python sim/augment.py demos data/raw --raw
if [ ! -f data/scripted_4/index.csv ]; then
  for s in 1 2 3 4; do
    python sim/augment.py demos data/scripted_$s --per-demo 15 --seed $s --scripted > scripted_$s.log 2>&1 &
  done
  wait
fi

echo "== 2/4 training"
train() { out=$1; shift; [ -f "$out/train_log.json" ] || python train_act.py "$@" --out "$out"; }
train runs/act_aug      $AUG
train runs/jepa_aug     $AUG --jepa
train runs/act_scripted $SCR
train runs/act_raw      data/raw --epochs 300

echo "== 3/4 testing on 100 unseen layouts (same layouts for every policy)"
evalp() { [ -f "$1/eval.json" ] || python sim/eval_policy.py "$1/policy.pt" --episodes 100 --videos "${2:-0}"; }
evalp runs/act_aug 3
evalp runs/jepa_aug 3
evalp runs/act_scripted
evalp runs/act_raw

echo "== 4/4 world-model probe"
[ -f runs/act_aug/probe.json ]  || python probe.py runs/act_aug/policy.pt  $AUG
[ -f runs/jepa_aug/probe.json ] || python probe.py runs/jepa_aug/policy.pt $AUG

python summarize.py
