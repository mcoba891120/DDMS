import torch
from typing import Dict, List, Optional
from ._gnn import RVEGNN

class RVEGNNEnsemble:
	def __init__(self, hyperparams:Dict, n_members:int=5, seed:int=0):
		self.hyperparams = hyperparams
		self.members: List[RVEGNN] = []
		for i in range(n_members):
			torch.manual_seed(seed+i)
			self.members.append(RVEGNN(**hyperparams))

	def load_state_dicts(self, state_dicts: List[dict]) -> 'RVEGNNEnsemble':
		assert len(state_dicts)==len(self.members), \
			f'expected {len(self.members)} state_dicts, got {len(state_dicts)}'
		for member, state_dict in zip(self.members, state_dicts):
			member.load_state_dict(state_dict)
		return self

	def to(self, device) -> 'RVEGNNEnsemble':
		for member in self.members:
			member.to(device)
		return self

	def eval(self) -> 'RVEGNNEnsemble':
		for member in self.members:
			member.eval()
		return self

	@torch.no_grad()
	def predict(self, data, mc_samples:Optional[int]=None):
		""" mean/std of `data.y` prediction across ensemble members (and MC samples, if given) """
		member_preds = []
		for member in self.members:
			if mc_samples:
				pred, _ = member.predict_mc(data, n_samples=mc_samples)
			else:
				pred, _ = member.mean_var(member(data.x, data.edge_index, data.edge_attr, data.volume, data.batch))
			member_preds.append(pred)

		preds = torch.stack(member_preds, dim=0)
		return preds.mean(0), preds.std(0)
