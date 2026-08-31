import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, SiLU, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MultiAggregation
from typing import List, Optional, Union

class GrainConv(MessagePassing):
	def __init__(self, in_ch, edge_ch, out_ch, aggr:Union[str,List[str]]='mean', dropout:float=0.0):
		n_aggr = len(aggr) if isinstance(aggr, list) else 1
		super().__init__(aggr=MultiAggregation(aggr) if isinstance(aggr, list) else aggr)

		self.message_mlp = Sequential(
			Linear(2*in_ch+edge_ch, out_ch), SiLU(),
			Linear(out_ch, out_ch),
		)
		self.message_dropout = Dropout(dropout)
		self.update_mlp = Sequential(
			Linear(in_ch+out_ch*n_aggr, out_ch), SiLU(),
			Linear(out_ch, out_ch),
		)
		self.update_dropout = Dropout(dropout)
		self.skip = Linear(in_ch, out_ch) if in_ch!=out_ch else None

	def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
		"""
		- x.size() == (num_nodes, in_ch)
		- edge_index.size() == (2, num_edges)
		- edge_attr.size() == (num_edges, edge_ch)
		- returns (num_nodes, out_ch)
		"""
		return self.propagate(edge_index, x=x, edge_attr=edge_attr)

	def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
		m = self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
		return self.message_dropout(m)

	def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
		out = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
		out = self.update_dropout(out)
		skip = x if self.skip is None else self.skip(x)
		return out + skip
