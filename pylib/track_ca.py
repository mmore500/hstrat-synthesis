import tqdm
import numpy as np 
from downstream import dstream


algo = dstream.steady_algo


S = 64

data = np.load("hist.npy").astype(np.bool_)# [:, 180:300, 240:400]

H, W = data.shape[1:]
print(H, W)
hst_markers = np.random.randint(0, 2, size=(H, W, S), dtype=np.bool_)  # deposit random stuff everywhere first gen

ancestor_search = [
    (i, j) for i in range(-1, 2) for j in range(-1, 2)
]

def get_slices(x, y, H, W):
    x_slice = slice(0, W + x) if x < 0 else slice(x, W)
    y_slice = slice(0, H + y) if y < 0 else slice(y, H)
    return y_slice, x_slice

parent_idx = np.arange(0, H * W).reshape(H, W)
parent_keys = []
for x, y in ancestor_search:
    arr = np.zeros((H, W), dtype=np.int64)
    arr[*get_slices(-x, -y, H, W)] = parent_idx[*get_slices(x, y, H, W)]
    parent_keys.append(arr[None])
parents = np.concatenate(parent_keys)

for t in tqdm.trange(1, data.shape[0]):
    bit_to_assign = algo.assign_storage_site(S, t)
    parent = data[t - 1]
    curr = data[t]

    ancestor_matrices = []
    for x, y in ancestor_search:
        arr = np.zeros_like(curr, dtype=np.int64)
        arr[*get_slices(-x, -y, H, W)] = parent[*get_slices(x, y, H, W)]
        ancestor_matrices.append(arr[None])
    scores = np.concatenate(ancestor_matrices)

    winning_scores = parents.transpose(1, 2, 0)[np.eye(scores.shape[0], dtype=np.bool_)[scores.argmax(axis=0)]].reshape(H, W)
    hst_markers[:] = hst_markers[winning_scores // W, winning_scores % W]
    if bit_to_assign is not None:
        hst_markers[:, :, bit_to_assign] = np.random.randint(0, 2, size=(H, W), dtype=np.bool_)

hst_markers[~curr] = 0

genome = np.packbits(hst_markers, axis=-1, bitorder='big').view(dtype=">u8")
assert genome.shape[-1] == 1

np.save("genome.npy", genome)
extant = []
for i in genome[genome != 0]:
    extant.append(np.asarray(t, dtype=np.uint32).astype(">u4").tobytes().hex() + i.tobytes().hex())
print(extant)

import pandas as pd


