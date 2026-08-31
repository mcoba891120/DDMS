import torch
import torch.nn as nn
from typing import List, Union
from ._base import BaseModule
from .layer._gnnLayer import GrainConv
from .layer._gnnPool import GraphPool
from ..loss import MAPE, Correlation

class RVEGNN(BaseModule):
	def __init__(self, in_ch, edge_ch, hid_ch, out_ch, n_layer,
				 aggr:Union[str,List[str]]='mean', pool:str='volume_weighted',
				 dropout:float=0.0, heteroscedastic:bool=False):
		super().__init__()
		self.out_ch = out_ch
		self.heteroscedastic = heteroscedastic

		self.encoder = nn.Linear(in_ch, hid_ch)
		self.convs = nn.ModuleList([
			GrainConv(hid_ch, edge_ch, hid_ch, aggr=aggr, dropout=dropout) for _ in range(n_layer)
		])
		self.act = nn.SiLU()
		self.pool = GraphPool(hid_ch, mode=pool)
		self.head_dropout = nn.Dropout(dropout)
		self.head = nn.Sequential(
			nn.Linear(self.pool.out_ch, hid_ch), nn.SiLU(),
			nn.Linear(hid_ch, out_ch*2 if heteroscedastic else out_ch),
		)

	def configure_loss(self) -> dict:
		criterion = {'mae': nn.L1Loss(), 'mape': MAPE(), 'corr': Correlation()}
		if self.heteroscedastic:
			criterion['nll'] = nn.GaussianNLLLoss()
		return {
			'criterion': criterion,
			'desc': 'gaussian_nll' if self.heteroscedastic else 'mae',
		}

	def configure_optimizer(self, lr, num_epochs=None, num_batches_per_epoch=None) -> dict:
		optimizer = torch.optim.Adam(self.parameters(), lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, factor=0.5, patience=50, min_lr=1e-5,
		)
		return {
			'optimizers': {'opt': optimizer},
			'schedulers': {'sch': scheduler},
			'desc': 'Adam with ReduceOnPlateau',
		}

	def forward(self, x, edge_index, edge_attr, volume, batch):
		"""
		- x.size() == (num_nodes, in_ch)
		- edge_index.size() == (2, num_edges)
		- edge_attr.size() == (num_edges, edge_ch)
		- volume.size() == (num_nodes, 1)
		- batch.size() == (num_nodes,)
		- returns (num_graphs, out_ch), or (num_graphs, 2*out_ch) if `heteroscedastic`
		"""
		h = self.encoder(x)
		for conv in self.convs:
			h = self.act(conv(h, edge_index, edge_attr))

		pooled = self.pool(h, volume, batch)
		return self.head(self.head_dropout(pooled))

	def mean_var(self, y_raw):
		""" split head output into (mean, var); var is 1 if not `heteroscedastic` """
		if not self.heteroscedastic:
			return y_raw, torch.ones_like(y_raw)
		mean, log_var = y_raw.chunk(2, dim=-1)
		return mean, log_var.exp()

	def step_forward(self, data, criterion: dict) -> dict:
		y_raw = self(data.x, data.edge_index, data.edge_attr, data.volume, data.batch)
		mean, var = self.mean_var(y_raw)

		if self.heteroscedastic:
			loss = criterion.get('nll')(mean, data.y, var)
		else:
			loss = criterion.get('mae')(mean, data.y)

		mape = criterion.get('mape')(mean, data.y)
		acc = 1 - mape
		corr = criterion.get('corr')(mean, data.y, return_avg=False).mean()

		return {
			'loss': loss,
			'mape': mape,
			'accuracy': acc if acc>=0 else torch.zeros(1),
			'corr': corr,
		}

	def step_backward(self, kwloss: dict, optimizers: dict) -> None:
		opt = optimizers.get('opt')
		opt.zero_grad()
		kwloss.get('loss').backward()
		opt.step()

	@torch.no_grad()
	def predict_mc(self, data, n_samples:int=30):
		""" MC-dropout: mean/std over `n_samples` stochastic forward passes """
		was_training = self.training
		self.train()
		try:
			preds = torch.stack([
				self.mean_var(self(data.x, data.edge_index, data.edge_attr, data.volume, data.batch))[0]
				for _ in range(n_samples)
			], dim=0)
		finally:
			self.train(was_training)

		return preds.mean(0), preds.std(0)
