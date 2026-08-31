"""Tests for `RVEGNN`, `RVEGNNEnsemble`, and `MemoryDataset_GNN`."""
import copy

import torch
import pytest
from torch_geometric.data import Data, Batch

from ddms.surrogate.model import RVEGNN, RVEGNNEnsemble
from ddms.surrogate.dataset import MemoryDataset_GNN


def _make_graph(num_nodes, num_edges, out_ch, seed):
	"""Build one RVE microstructure graph with the same node/edge feature
	layout as `MemoryDataset_GNN._get_data` produces."""
	g = torch.Generator().manual_seed(seed)
	x = torch.randn(num_nodes, 5, generator=g)			# euler(3) + volume(1) + num_neighbor(1)
	volume = x[:, 3:4].abs() + 0.1							# keep strictly positive
	edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=g)
	edge_attr = torch.randn(num_edges, 5, generator=g)		# normal(3) + misorientation(1) + surfaceArea(1)
	y = torch.randn(1, out_ch, generator=g)
	return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, volume=volume, y=y)


def _make_batch(out_ch=3):
	graphs = [
		_make_graph(num_nodes=4, num_edges=6, out_ch=out_ch, seed=0),
		_make_graph(num_nodes=3, num_edges=4, out_ch=out_ch, seed=1),
	]
	return Batch.from_data_list(graphs)


def test_rve_gnn_forward_shape():
	batch = _make_batch(out_ch=3)
	model = RVEGNN(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2)

	y_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.volume, batch.batch)

	assert y_pred.shape == (2, 3)
	assert torch.isfinite(y_pred).all()


def test_rve_gnn_train_step_updates_parameters():
	batch = _make_batch(out_ch=3)
	model = RVEGNN(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2)

	criterion = model.configure_loss()['criterion']
	optimizers = model.configure_optimizer(lr=1e-2)['optimizers']

	before = copy.deepcopy(list(model.parameters()))

	kwloss = model.step_forward(batch, criterion)
	assert torch.isfinite(kwloss['loss'])

	model.step_backward(kwloss, optimizers)

	after = list(model.parameters())
	assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_memory_dataset_gnn_loads_graphs(tmp_path):
	num_graphs = 2
	out_ch = 2

	for i in range(num_graphs):
		folder = tmp_path / 'raw' / f'rve_{i}'
		folder.mkdir(parents=True)

		num_nodes, seq_len = 5, 3
		graph = Data(
			euler=torch.randn(seq_len, num_nodes, 3),
			volume=torch.rand(num_nodes, 1) + 0.1,
			num_neighbor=torch.randint(3, 8, (num_nodes, 1)).float(),
			edge_index=torch.randint(0, num_nodes, (2, 8)),
			edge_attr=torch.randn(8, 5),
		)
		torch.save(graph, folder / 'data.pt')
		torch.save(torch.randn(out_ch), folder / 'label.pt')

	dataset = MemoryDataset_GNN(str(tmp_path))

	assert len(dataset) == num_graphs
	assert dataset.data.x.size(-1) == 5		# euler(3) + volume(1) + num_neighbor(1)
	assert dataset.data.y.size(-1) == out_ch


@pytest.mark.parametrize('aggr,pool', [
	('mean', 'volume_weighted'),
	(['mean', 'max', 'std'], 'volume_weighted'),			# PNA-style multi-aggregation
	('mean', 'attention'),
	('mean', 'volume_weighted+max'),						# concatenated multi-pool
	(['mean', 'max'], 'attention+mean'),
])
def test_rve_gnn_aggregation_and_pooling_variants(aggr, pool):
	batch = _make_batch(out_ch=3)
	model = RVEGNN(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2, aggr=aggr, pool=pool)

	y_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.volume, batch.batch)

	assert y_pred.shape == (2, 3)
	assert torch.isfinite(y_pred).all()


def test_rve_gnn_heteroscedastic_head_and_nll_loss():
	batch = _make_batch(out_ch=3)
	model = RVEGNN(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2, heteroscedastic=True)

	y_raw = model(batch.x, batch.edge_index, batch.edge_attr, batch.volume, batch.batch)
	assert y_raw.shape == (2, 6)		# [mean(3), log_var(3)]

	mean, var = model.mean_var(y_raw)
	assert mean.shape == (2, 3)
	assert (var > 0).all()				# exp(log_var) must be strictly positive

	criterion = model.configure_loss()['criterion']
	optimizers = model.configure_optimizer(lr=1e-2)['optimizers']
	kwloss = model.step_forward(batch, criterion)
	assert torch.isfinite(kwloss['loss'])
	model.step_backward(kwloss, optimizers)		# should not raise


def test_rve_gnn_mc_dropout_gives_nonzero_spread_only_with_dropout():
	batch = _make_batch(out_ch=3)

	model = RVEGNN(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2, dropout=0.5)
	model.eval()					# the realistic case: predicting with a trained, eval()'d model
	mean, std = model.predict_mc(batch, n_samples=20)
	assert mean.shape == (2, 3)
	assert (std > 0).any()
	assert not model.training		# predict_mc must restore the caller's eval mode afterwards

	model_no_dropout = RVEGNN(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2, dropout=0.0)
	mean0, std0 = model_no_dropout.predict_mc(batch, n_samples=5)
	assert torch.allclose(std0, torch.zeros_like(std0))


def test_rve_gnn_ensemble_predict_shape_and_spread():
	batch = _make_batch(out_ch=3)
	hyperparams = dict(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2)

	ensemble = RVEGNNEnsemble(hyperparams, n_members=4, seed=42)
	mean, std = ensemble.predict(batch)

	assert mean.shape == (2, 3)
	assert (std > 0).any()		# independently-initialized members should disagree


def test_rve_gnn_ensemble_load_state_dicts_rejects_wrong_count():
	hyperparams = dict(in_ch=5, edge_ch=5, hid_ch=8, out_ch=3, n_layer=2)
	ensemble = RVEGNNEnsemble(hyperparams, n_members=3, seed=0)
	with pytest.raises(AssertionError):
		ensemble.load_state_dicts([{}, {}])		# wrong number of state_dicts


def test_memory_dataset_gnn_skips_incomplete_folders(tmp_path):
	complete = tmp_path / 'raw' / 'rve_complete'
	complete.mkdir(parents=True)
	graph = Data(
		euler=torch.randn(2, 4, 3),
		volume=torch.rand(4, 1) + 0.1,
		num_neighbor=torch.randint(3, 8, (4, 1)).float(),
		edge_index=torch.randint(0, 4, (2, 6)),
		edge_attr=torch.randn(6, 5),
	)
	torch.save(graph, complete / 'data.pt')
	torch.save(torch.randn(2), complete / 'label.pt')

	# missing label.pt entirely -> should be skipped, not raise
	incomplete = tmp_path / 'raw' / 'rve_incomplete'
	incomplete.mkdir(parents=True)
	torch.save(graph, incomplete / 'data.pt')

	dataset = MemoryDataset_GNN(str(tmp_path))

	assert len(dataset) == 1
