import re
import numpy as np
from typing import Callable, List, Tuple

def default_group_fn(desc: str) -> str:
	""" strip trailing `_<index>` from an RVE `desc`, e.g. 'morient_20_16_smooth_501' -> 'morient_20_16_smooth' """
	return re.sub(r'_\d+$', '', desc)

def _group_indices(dataset, group_fn: Callable[[str], str]) -> dict:
	groups = {}
	for i, desc in enumerate(dataset.data.desc):
		groups.setdefault(group_fn(desc), []).append(i)
	return groups

def stratified_split(dataset, train_split:float=0.7, seed:int=0,
					  group_fn:Callable[[str], str]=default_group_fn) -> Tuple[List[int], List[int]]:
	"""
	Args:
		dataset: a `MemoryDataset` whose collated `data.desc` holds one string per graph
		train_split: fraction of each group assigned to train
		seed: random seed
		group_fn: maps an RVE `desc` to its texture-group key
	Returns:
		train_idx, val_idx: index lists usable as `dataset[train_idx]`
	"""
	rng = np.random.RandomState(seed)
	groups = _group_indices(dataset, group_fn)

	train_idx, val_idx = [], []
	for idx in groups.values():
		idx = list(idx)
		rng.shuffle(idx)
		split = len(idx) if len(idx)<=1 else max(1, int(round(train_split*len(idx))))
		train_idx += idx[:split]
		val_idx += idx[split:]

	rng.shuffle(train_idx)
	rng.shuffle(val_idx)
	return train_idx, val_idx

def kfold_indices(dataset, k:int=5, seed:int=0,
				   group_fn:Callable[[str], str]=default_group_fn):
	"""
	Args:
		dataset: a `MemoryDataset` whose collated `data.desc` holds one string per graph
		k: number of folds
		seed: random seed
		group_fn: maps an RVE `desc` to its texture-group key
	Yields:
		(train_idx, val_idx) for each of the `k` folds
	"""
	rng = np.random.RandomState(seed)
	groups = _group_indices(dataset, group_fn)

	fold_of = {}
	for idx in groups.values():
		idx = list(idx)
		rng.shuffle(idx)
		for j, i in enumerate(idx):
			fold_of[i] = j % k

	all_idx = np.array(sorted(fold_of.keys()))
	fold_assignment = np.array([fold_of[i] for i in all_idx])

	for fold in range(k):
		val_idx = all_idx[fold_assignment==fold].tolist()
		train_idx = all_idx[fold_assignment!=fold].tolist()
		yield train_idx, val_idx
