"""Tests for `ddms.surrogate.validation` (texture-group-aware dataset splitting)."""
import torch
import pytest
from torch_geometric.data import Data

from ddms.surrogate.dataset import MemoryDataset_GNN
from ddms.surrogate.validation import default_group_fn, stratified_split, kfold_indices


def _write_rve_dataset(root, groups, realizations_per_group):
	for g in groups:
		for r in range(realizations_per_group):
			folder = root / 'raw' / f'{g}_{r}'
			folder.mkdir(parents=True)
			graph = Data(
				euler=torch.randn(2, 5, 3),
				volume=torch.rand(5, 1) + 0.1,
				num_neighbor=torch.randint(3, 8, (5, 1)).float(),
				edge_index=torch.randint(0, 5, (2, 8)),
				edge_attr=torch.randn(8, 5),
			)
			torch.save(graph, folder / 'data.pt')
			torch.save(torch.randn(2), folder / 'label.pt')
	return MemoryDataset_GNN(str(root))


def test_default_group_fn_strips_trailing_realization_index():
	assert default_group_fn('morient_20_16_smooth_501') == 'morient_20_16_smooth'
	assert default_group_fn('texture0_3') == 'texture0'


@pytest.fixture
def rve_dataset(tmp_path):
	return _write_rve_dataset(tmp_path, groups=['textureA', 'textureB', 'textureC'], realizations_per_group=4)


def test_stratified_split_covers_every_index_exactly_once(rve_dataset):
	train_idx, val_idx = stratified_split(rve_dataset, train_split=0.75, seed=0)

	assert set(train_idx).isdisjoint(val_idx)
	assert set(train_idx) | set(val_idx) == set(range(len(rve_dataset)))


def test_stratified_split_keeps_every_group_on_both_sides(rve_dataset):
	train_idx, val_idx = stratified_split(rve_dataset, train_split=0.75, seed=0)

	all_groups = {default_group_fn(d) for d in rve_dataset.data.desc}
	train_groups = {default_group_fn(rve_dataset.data.desc[i]) for i in train_idx}
	val_groups = {default_group_fn(rve_dataset.data.desc[i]) for i in val_idx}

	assert train_groups == all_groups
	assert val_groups == all_groups


def test_kfold_indices_partitions_dataset_without_overlap(rve_dataset):
	folds = list(kfold_indices(rve_dataset, k=4, seed=0))
	assert len(folds) == 4

	all_val_idx = []
	for train_idx, val_idx in folds:
		assert set(train_idx).isdisjoint(val_idx)
		assert set(train_idx) | set(val_idx) == set(range(len(rve_dataset)))
		all_val_idx += val_idx

	assert sorted(all_val_idx) == list(range(len(rve_dataset)))


def test_kfold_indices_spreads_each_group_across_folds(rve_dataset):
	folds = list(kfold_indices(rve_dataset, k=4, seed=0))

	for _, val_idx in folds:
		val_groups = {default_group_fn(rve_dataset.data.desc[i]) for i in val_idx}
		assert len(val_groups) > 1
