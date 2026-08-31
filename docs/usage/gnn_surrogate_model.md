# GNN surrogate model (RVEGNN)

`RVEGNN` predicts an RVE's homogenized mechanical response directly from its
microstructure graph: grains are nodes, grain boundaries are edges. Lives
alongside `LMSC` under `ddms.surrogate.model`.

## Graph representation

| | source | shape |
|---|---|---|
| node features `x` | initial grain orientation (Euler angles), grain volume, neighbor count | `(num_grains, 5)` |
| edge index | grain-boundary adjacency | `(2, num_edges)` |
| edge features `edge_attr` | boundary normal, misorientation angle, boundary area | `(num_edges, 5)` |
| target `y` | homogenized response (e.g. yield stress, or a fixed-length sampled stress-strain curve) | `(out_ch,)` |

`GrainConv` message-passing layers update grain embeddings from grain and
boundary features, `GraphPool` aggregates them into an RVE-level embedding,
an MLP head maps it to the target response.

## Preparing data

`MemoryDataset_GNN` expects, per RVE, a `{data_path}/raw/{name}/` folder containing:

- `data.pt` — the `torch_geometric.data.Data` graph produced by
  `HandlerDamask.createGraphDataset` (`-m genDataset`), with attributes
  `euler`, `volume`, `num_neighbor`, `edge_index`, `edge_attr`.
- `label.pt` — a `torch.Tensor` of shape `(out_ch,)` holding the target
  response for that RVE (e.g. `get_YS` from `ddms.numeric.processing`).

## Training

``` py
import sys
sys.path.append('../../python')
from ddms.surrogate import Trainer
from ddms.surrogate.model import RVEGNN
from ddms.surrogate.dataset import MemoryDataset_GNN
from ddms.surrogate.validation import stratified_split

trainer = Trainer(args)  # same Args/argparse setup as LMSC, see surrogate/main.py
trainer.init(
	Model=RVEGNN,
	Dataset=MemoryDataset_GNN,
	hyperparams=dict(in_ch=5, edge_ch=5, hid_ch=32, out_ch=1, n_layer=3),
	split_fn=stratified_split,
)
trainer.fit()
```

`in_ch`/`edge_ch` are fixed by the graph representation above; `hid_ch`,
`out_ch`, `n_layer` are free hyperparameters — `out_ch` matches whatever
you saved into `label.pt`.

## Aggregation and pooling

``` py
RVEGNN(
	in_ch=5, edge_ch=5, hid_ch=32, out_ch=1, n_layer=3,
	aggr=['mean', 'max', 'std'],        # PNA-style: concatenate several reducers
	pool='volume_weighted+attention',   # concatenate several pooling strategies
)
```

- `aggr`: `'mean'`/`'max'`/`'sum'`/`'std'`, or a list to concatenate several.
- `pool`: `'volume_weighted'` (default), `'mean'`, `'max'`, `'sum'`,
  `'attention'`, or several joined with `+`.

## Uncertainty quantification

``` py
# aleatoric: heteroscedastic head, Gaussian NLL loss
model = RVEGNN(..., heteroscedastic=True)

# epistemic: MC-dropout at inference (needs dropout>0 at training time)
model = RVEGNN(..., dropout=0.2)
mean, std = model.predict_mc(data, n_samples=30)

# epistemic: deep ensemble (train each member independently, then combine)
from ddms.surrogate.model import RVEGNNEnsemble
ensemble = RVEGNNEnsemble(hyperparams, n_members=5).load_state_dicts(state_dicts)
mean, std = ensemble.predict(data)
```

## Validation

`Trainer.init(..., split_fn=...)` overrides the default random train/val
split. `ddms.surrogate.validation`:

- `stratified_split`: single train/val split, splitting within each
  texture group.
- `kfold_indices`: stratified k-fold generator, yields `(train_idx, val_idx)`
  per fold.

Both group RVEs by `default_group_fn`, which strips a trailing `_<index>`
from `desc` (e.g. `'morient_20_16_smooth_501'` -> `'morient_20_16_smooth'`).
Pass your own `group_fn` if your naming convention differs.
