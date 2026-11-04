#!/usr/bin/env bash
# Extra runs that make the conclusions trustworthy (about 3 hours; fine to leave overnight).
#   bash run_extra.sh 2>&1 | tee logs/run_extra.log
set -e
AUG="data/aug_1 data/aug_2 data/aug_3 data/aug_4"
train() { out=$1; shift; [ -f "$out/train_log.json" ] || python train_act.py "$@" --out "$out"; }

# 1) no-human-data baseline with about the SAME number of episodes as yours (fair comparison)
train runs/act_scripted_half data/scripted_1 data/scripted_2
# 2) a second training seed for ACT and ACT-JEPA (is the difference real or luck?)
train runs/act_aug_s1  $AUG --seed 1
train runs/jepa_aug_s1 $AUG --jepa --seed 1

# 3) bigger test: 300 new layouts per policy (same 300 for every policy)
for r in act_aug jepa_aug act_aug_s1 jepa_aug_s1 act_scripted act_scripted_half act_raw; do
  mkdir -p runs/$r/eval300
  [ -f runs/$r/eval300/eval.json ] || \
    python sim/eval_policy.py runs/$r/policy.pt --episodes 300 --seed 777 --out runs/$r/eval300
done
[ -f runs/act_aug_s1/probe.json ]  || python probe.py runs/act_aug_s1/policy.pt  $AUG
[ -f runs/jepa_aug_s1/probe.json ] || python probe.py runs/jepa_aug_s1/policy.pt $AUG
python summarize.py
