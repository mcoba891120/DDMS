import torch 
from torch import Tensor
from torch.nn import Module, Linear, Sequential
from typing import Optional

class LMSCLayer(Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.a = Linear(in_ch, out_ch)
		self.b = Linear(in_ch, out_ch)
	
	def forward(self, l):
		return torch.tanh(self.a(l)) * torch.tanh(self.b(l))

class LMSCCell(Module):
	def __init__(self, in_ch, hid_ch, stt_ch, n_layer) -> None:
		super().__init__()
		self.stt_ch = stt_ch

		self.dnn = []
		for l in range(n_layer):
			i, o = (in_ch+stt_ch, hid_ch) if l==0 else (hid_ch, hid_ch)
			self.dnn.append(LMSCLayer(i, o))
		self.dnn = Sequential(*self.dnn)
		self.alpha = Linear(hid_ch, stt_ch)
		self.beta = Linear(hid_ch, stt_ch)
	
	def forward(self, x, x_norm:Optional[Tensor]=None, chi_0:Optional[Tensor]=None):
		"""
		- x.size() == (b, in_ch)
		- x_norm.size() == (b, in_ch)
		- chi_0.size() == (b, stt_ch)
		"""
		if x_norm is None:
			# to prevent using `torch.sqrt()` in torchscript 
			# x_norm = x.square().sum(dim=-1, keepdim=True).sqrt().clamp(1e-15, 100)			
			x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(1e-15, 100)

		if chi_0 is None:
			chi_0 = x.new_zeros(x.size(0), self.stt_ch)

		# >--------------------------------------------------
		# add noise to chi_0
		# lmb2 = 3e-4
		# noise = torch.rand(*chi_0.size(), device=chi_0.device)*2*lmb2 - lmb2
		# chi_0_ = chi_0 + noise
		# >--------------------------------------------------
		chi_0_ = chi_0
		# >--------------------------------------------------

		l = self.dnn(torch.cat([x/x_norm, chi_0_], dim=-1))
		alpha = torch.exp(self.alpha(l))
		beta = torch.tanh(self.beta(l))
		chi_1 = torch.exp(-x_norm*alpha) * (chi_0_-beta) + beta
		return chi_1, alpha, beta

