"""Warm-start checkpoint for a task whose critic gained inputs appended at the end: the source
checkpoint with the critic's first layer padded by zero-weight columns (normalizer mean 0,
var/std 1) and Adam's moments padded with zeros, so the critic starts exactly as before and
ignores the new inputs until it learns them. Everything else is copied.

    uv run python scripts/make_padded_init.py <src.pt> <dst.pt> <new critic input dim>
"""

import sys

import torch

src, dst, n_new = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(src, map_location="cpu", weights_only=False)
cs = ck["critic_state_dict"]
n_old = cs["mlp.0.weight"].shape[1]
k = n_new - n_old
assert k > 0, (n_old, n_new)


def pad(t, value=0.0):
    return torch.cat([t, torch.full((*t.shape[:-1], k), value, dtype=t.dtype)], dim=-1)


for key in list(cs):
    if key in ("mlp.0.weight", "obs_normalizer._mean"):
        cs[key] = pad(cs[key])
    elif key in ("obs_normalizer._var", "obs_normalizer._std"):
        cs[key] = pad(cs[key], 1.0)
n_opt = 0
for state in ck["optimizer_state_dict"]["state"].values():
    for name, t in state.items():
        if torch.is_tensor(t) and t.dim() == 2 and t.shape == (cs["mlp.0.weight"].shape[0], n_old):
            state[name] = pad(t)
            n_opt += 1
torch.save(ck, dst)
print(
    f"critic input {n_old} -> {n_new}; padded critic layer, normalizer and {n_opt} optimizer moments"
)
