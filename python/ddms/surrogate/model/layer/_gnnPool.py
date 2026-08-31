import torch
from torch import Tensor
from torch.nn import Module, Linear
from torch_geometric.utils import scatter, softmax

class GraphPool(Module):
	def __init__(self, in_ch, mode:str='volume_weighted'):
		super().__init__()
		self.modes = mode.split('+')
		for m in self.modes:
			if m not in ('volume_weighted', 'mean', 'max', 'sum', 'attention'):
				raise ValueError(f'unknown pool mode: {m}')
		if 'attention' in self.modes:
			self.att = Linear(in_ch, 1)
		self.out_ch = in_ch*len(self.modes)

	def forward(self, h: Tensor, volume: Tensor, batch: Tensor) -> Tensor:
		"""
		- h.size() == (num_nodes, in_ch)
		- volume.size() == (num_nodes, 1)
		- batch.size() == (num_nodes,)
		- returns (num_graphs, out_ch)
		"""
		outs = []
		for m in self.modes:
			if m=='volume_weighted':
				total_volume = scatter(volume, batch, dim=0, reduce='sum')[batch]
				weight = volume / total_volume.clamp_min(1e-12)
				outs.append(scatter(h*weight, batch, dim=0, reduce='sum'))
			elif m=='attention':
				score = softmax(self.att(h), batch)
				outs.append(scatter(h*score, batch, dim=0, reduce='sum'))
			else:
				outs.append(scatter(h, batch, dim=0, reduce=m))

		return torch.cat(outs, dim=-1)
