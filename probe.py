#!/usr/bin/env python3
"""World-model probe, as in the ACT-JEPA paper (Sec. 4.3).

Freeze a trained policy's encoder. Train a fresh, small decoder to predict the
robot's next CHUNK joint states from the frozen representation. Lower error on the
held-out episodes = the representation knows more about how things will evolve.
Reports RMSE and ATE (mean distance between predicted and true joint trajectories),
in radians.

  python probe.py runs/act_aug/policy.pt data/aug_1 data/aug_2 data/aug_3 data/aug_4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_act import ACT, CHUNK, ChunkDataset, load_episodes, split

J = 17   # joint part of the observation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("data", nargs="+")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    model = ACT(**cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    stats = {k: np.array(v, dtype=np.float32) for k, v in ck["stats"].items()}
    tr_eps, val_eps = split(load_episodes(args.data), ck.get("seed", 0))   # same split as training
    tr, va = ChunkDataset(tr_eps, stats), ChunkDataset(val_eps, stats)
    dl = torch.utils.data.DataLoader(tr, 256, shuffle=True, drop_last=True)
    vl = torch.utils.data.DataLoader(va, 256)
    d = model.act_q.shape[-1]
    torch.manual_seed(0)
    q = nn.Parameter(torch.randn(1, CHUNK, d, device=dev) * 0.02)
    dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 2 * d, 0.0, batch_first=True),
                                1).to(dev)
    head = nn.Linear(d, J).to(dev)
    params = [q] + list(dec.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)
    std = torch.tensor(stats["obs_std"][:J], device=dev)

    def run(o):
        mem = model.encoder(o)
        return head(dec(q.expand(o.shape[0], -1, -1), mem))

    step = 0
    for ep in range(args.epochs):
        for o, _, fut in dl:
            o, fut = o.to(dev), fut.to(dev)
            loss = ((run(o) - fut[..., :J]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    se, ate, n = 0.0, 0.0, 0
    with torch.no_grad():
        for o, _, fut in vl:
            o, fut = o.to(dev), fut.to(dev)
            err = (run(o) - fut[..., :J]) * std          # back to radians
            se += (err ** 2).mean(-1).sum().item()
            ate += err.norm(dim=-1).mean(-1).sum().item()
            n += err.shape[0]
    rmse = float(np.sqrt(se / (n * CHUNK)))
    ate = float(ate / n)
    name = "ACT-JEPA" if cfg.get("jepa") else "ACT"
    print(f"{name}: future-joint RMSE {rmse:.4f} rad, ATE {ate:.4f} rad  ({n} held-out samples)")
    json.dump(dict(model=name, rmse=rmse, ate=ate, samples=n),
              open(os.path.join(os.path.dirname(args.policy), "probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
