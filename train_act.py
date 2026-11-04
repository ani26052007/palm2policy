#!/usr/bin/env python3
"""Train ACT or ACT-JEPA on the robot episodes (state-based).

ACT (Zhao et al. 2023): from the current observation, predict the next CHUNK actions.
  Like the ACT baseline in the ACT-JEPA paper, we use the plain autoencoder variant
  (no CVAE).
ACT-JEPA (Vujinovic & Kovacevic): same network, plus a second training objective.
  A predictor must guess the abstract representations of the next CHUNK observations.
  The targets come from a target encoder: an exponential-moving-average copy of the
  context encoder that reads the future observation sequence. The loss is
  L1(actions) + L1(latents). At test time only the encoder and action decoder run,
  so it is exactly as fast as ACT.
  The paper's targets are proprioception only (--target proprio). Our default
  (--target full) also predicts future touch, tub pose and notebook, which the
  paper lists as untried.

  python train_act.py data/aug_1 data/aug_2 --out runs/act_aug
  python train_act.py data/aug_1 data/aug_2 --out runs/jepa_aug --jepa
"""
import argparse
import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

CHUNK = 20          # 20 steps = 0.67 s at 30 Hz
# observation layout (see sim/augment.py): one token per kind of sensing
PARTS = [(0, 17), (17, 23), (23, 27), (27, 31)]   # joints, tub, notebook, touch
PROPRIO = [0]


def load_episodes(dirs):
    eps = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "episodes", "*.npz"))):
            e = np.load(f)
            eps.append((e["obs"].astype(np.float32), e["act"].astype(np.float32), str(e["source"])))
    return eps


def split(eps, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eps))
    n_val = max(1, len(eps) // 10)
    return [eps[i] for i in order[n_val:]], [eps[i] for i in order[:n_val]]


def compute_stats(eps):
    O = np.concatenate([e[0] for e in eps])
    A = np.concatenate([e[1] for e in eps])
    return dict(obs_mean=O.mean(0), obs_std=np.maximum(O.std(0), 1e-2),
                act_mean=A.mean(0), act_std=np.maximum(A.std(0), 1e-2))


def pad_to(x, n):
    if len(x) == 0:
        return None
    return x if len(x) >= n else np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)])


class ChunkDataset(torch.utils.data.Dataset):
    """(obs_t, actions t..t+K-1, observations t+1..t+K)."""

    def __init__(self, eps, stats, chunk=CHUNK):
        self.chunk = chunk
        self.obs = [(o - stats["obs_mean"]) / stats["obs_std"] for o, _, _ in eps]
        self.act = [(a - stats["act_mean"]) / stats["act_std"] for _, a, _ in eps]
        self.index = [(i, t) for i, o in enumerate(self.obs) for t in range(len(o))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        o = self.obs[i]
        a = pad_to(self.act[i][t:t + self.chunk], self.chunk)
        fut = o[t + 1:t + 1 + self.chunk]
        fut = pad_to(fut if len(fut) else o[-1:], self.chunk)
        return torch.from_numpy(o[t]), torch.from_numpy(a), torch.from_numpy(fut)


def sinusoid(n, d):
    pos = torch.arange(n)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Encoder(nn.Module):
    """One token per kind of sensing -> transformer. Also encodes observation sequences
    (used as the JEPA target encoder): tokens get a time encoding, then are pooled per step."""

    def __init__(self, d, heads, layers):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(b - a, d) for a, b in PARTS])
        self.kind = nn.Parameter(torch.randn(len(PARTS), d) * 0.02)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True), layers)

    def tokens(self, x, parts):
        return torch.stack([self.proj[p](x[..., a:b]) for p, (a, b) in
                            ((p, PARTS[p]) for p in parts)], -2) + self.kind[parts]

    def forward(self, x):                                   # (B, obs) -> (B, P, d)
        return self.tf(self.tokens(x, list(range(len(PARTS)))))

    def encode_sequence(self, x, time_pe, parts):           # (B, T, obs) -> (B, T, d)
        B, T, _ = x.shape
        t = self.tokens(x, parts) + time_pe[:T][None, :, None, :]
        h = self.tf(t.reshape(B, T * len(parts), -1))
        return h.reshape(B, T, len(parts), -1).mean(2)


class ACT(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk=CHUNK, d=256, heads=8, enc=4, dec=4,
                 jepa=False, target="full"):
        super().__init__()
        self.chunk, self.jepa = chunk, jepa
        self.target_parts = PROPRIO if target == "proprio" else list(range(len(PARTS)))
        self.encoder = Encoder(d, heads, enc)
        dec_layer = lambda: nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True)
        self.act_q = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.act_dec = nn.TransformerDecoder(dec_layer(), dec)
        self.act_head = nn.Linear(d, act_dim)
        if jepa:
            self.pred_q = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
            self.pred_dec = nn.TransformerDecoder(dec_layer(), dec)
            self.pred_head = nn.Linear(d, d)
            self.register_buffer("time_pe", sinusoid(chunk + 1, d))

    def forward(self, obs):
        mem = self.encoder(obs)
        q = self.act_q.expand(obs.shape[0], -1, -1)
        return self.act_head(self.act_dec(q, mem)), mem

    def predict_latents(self, mem):
        q = self.pred_q.expand(mem.shape[0], -1, -1)
        return self.pred_head(self.pred_dec(q, mem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jepa", action="store_true", help="train ACT-JEPA instead of ACT")
    ap.add_argument("--target", choices=["full", "proprio"], default="full")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--small", action="store_true", help="smaller model for CPU tests")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    eps = load_episodes(args.data)
    tr_eps, val_eps = split(eps, args.seed)
    stats = compute_stats(tr_eps)
    tr, va = ChunkDataset(tr_eps, stats), ChunkDataset(val_eps, stats)
    dl = torch.utils.data.DataLoader(tr, args.batch, shuffle=True, drop_last=len(tr) > args.batch)
    vl = torch.utils.data.DataLoader(va, args.batch, shuffle=False)
    obs_dim, act_dim = tr_eps[0][0].shape[1], tr_eps[0][1].shape[1]
    size = dict(d=128, heads=4, enc=2, dec=2) if args.small else {}
    cfg = dict(obs_dim=obs_dim, act_dim=act_dim, jepa=args.jepa, target=args.target, **size)
    model = ACT(**cfg).to(dev)
    target_enc = None
    if args.jepa:
        target_enc = copy.deepcopy(model.encoder).eval()
        for p in target_enc.parameters():
            p.requires_grad_(False)
    n_par = sum(p.numel() for n, p in model.named_parameters())
    print(f"{'ACT-JEPA (' + args.target + ' targets)' if args.jepa else 'ACT'} | "
          f"{len(tr_eps)} train / {len(val_eps)} val episodes, {len(tr)} samples, "
          f"{n_par/1e6:.1f}M params, device {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    log, best, step, t0 = [], 1e9, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        la, lo = [], []
        for o, a, fut in dl:
            o, a, fut = o.to(dev), a.to(dev), fut.to(dev)
            pred, mem = model(o)
            l_act = (pred - a).abs().mean()
            loss = l_act
            if args.jepa:
                with torch.no_grad():
                    tgt = target_enc.encode_sequence(fut, model.time_pe, model.target_parts)
                l_obs = (model.predict_latents(mem) - tgt).abs().mean()
                loss = loss + l_obs
                lo.append(l_obs.item())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if target_enc is not None:
                with torch.no_grad():
                    for pt, pc in zip(target_enc.parameters(), model.encoder.parameters()):
                        pt.mul_(args.ema).add_(pc.detach(), alpha=1 - args.ema)
            la.append(l_act.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        model.eval()
        with torch.no_grad():
            vl1 = float(np.mean([(model(o.to(dev))[0] - a.to(dev)).abs().mean().item()
                                 for o, a, _ in vl]))
        row = dict(epoch=ep + 1, train_action_l1=float(np.mean(la)), val_action_l1=vl1)
        if lo:
            row["train_latent_l1"] = float(np.mean(lo))
        log.append(row)
        extra = f"  latent L1 {np.mean(lo):.4f}" if lo else ""
        print(f"epoch {ep+1:3d}  train L1 {np.mean(la):.4f}  val L1 {vl1:.4f}{extra}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl1 < best:
            best = vl1
            torch.save(dict(model=model.state_dict(), cfg=cfg, seed=args.seed, data=args.data,
                            stats={k: v.tolist() for k, v in stats.items()}),
                       os.path.join(args.out, "policy.pt"))
        if args.max_steps and step >= args.max_steps:
            break
    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
    print(f"best val action L1 {best:.4f} -> {args.out}/policy.pt")


if __name__ == "__main__":
    main()
