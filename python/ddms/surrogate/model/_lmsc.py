import torch 
import torch.nn as nn
from torch import Tensor 
from ranger21 import Ranger21
from typing import Optional
from ._base import BaseModule
from .layer._lmscCell import LMSCCell
from ..loss import MAPE, MAEAlpha, Correlation
from ..mechanics import dCSdHS_to_DDSDDE
from ..tensor import dev5_to_dev6, dev5_to_m6, m6_to_dev5

class LMSC(BaseModule):
	def __init__(self, in_ch, hid_ch, stt_ch, out_ch, n_layer, tilda_alpha=1e10):
		super().__init__()
		self.tilda_alpha = tilda_alpha

		self.lmsc = LMSCCell(in_ch, hid_ch, stt_ch, n_layer)
		self.out = nn.Linear(stt_ch, out_ch, bias=False)

	def configure_loss(self) -> dict:
		return {
			'criterion': {
				'mae': nn.L1Loss(), 'mape': MAPE(), 
				'corr': Correlation(), 'maeA': MAEAlpha(tilda_alpha=self.tilda_alpha)
			}, 
			'desc': 'mae / mape / corr'
		}

	def configure_optimizer(self, lr, num_epochs=None, num_batches_per_epoch=None) -> dict:
		# optimizer = torch.optim.Adam(self.parameters(), lr)
		optimizer = Ranger21(self.parameters(), lr, 
			num_epochs=num_epochs, num_batches_per_epoch=num_batches_per_epoch, 
			betas=(.9, 0.999), eps=1.0e-7, use_madgrad=True
		)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, factor=0.5, patience=50, min_lr=1e-4, verbose=True
		)
		return {
			'optimizers': {'opt': optimizer},
			'schedulers': {'sch': scheduler},
			'desc': 'Ranger21 with ReduceOnPlateau',
		}

	def forward(self, x, x_norm:Optional[Tensor]=None, chi:Optional[Tensor]=None):
		"""
		- x.size() == (b, s, in_ch)
		- x_norm.size() == (b, in_ch)
		- chi.size() == (b, stt_ch)
		"""
		chis = []
		alphas = []
		betas = []
		for s in range(x.size(1)):
			chi, alpha, beta = self.lmsc(x[:,s,:], x_norm, chi)
			chis.append(chi)
			alphas.append(alpha)
			betas.append(beta)
		chis = torch.stack(chis, dim=1)		# (b, s, c)
		alphas = torch.stack(alphas, dim=1)	# (b, s, c)
		betas = torch.stack(betas, dim=1)	# (b, s, c)
		y = self.out(chis)
		return y, chi, alphas, betas
	
	def forward_J(self, data, scaler):
		# for conistent tangent jacobian calculation
		K = 7.4972e+10
		eng = torch.tensor([1,1,1,2,2,2], device=data.dHS.device)
		norm = torch.tensor([1,1,1,0,0,0], device=data.dHS.device)

		# TODO: recursively calc. J
		y_pred = []
		J_pred = []
		alphas = []
		for s in range(data.dHS.size(1)):
			dHS = data.dHS[:,s:s+1,:].mul(eng).requires_grad_()								# convert to engineering shear component
			dHS_ = dHS.div(eng)
			dHSdev5 = m6_to_dev5(dHS_)														# (b, 1, 5)
			y_pred_, chi, alpha, _ = self.forward(dHSdev5, chi=None if s==0 else chi)		# (b, 1, 5)
			y_pred.append(y_pred_)
			alphas.append(alpha)

			if y_pred_.requires_grad:
				# deviatoric 
				CSdev5 = scaler.inverse_transform(y_pred_, 'CSdev5')						# (b, 1, 5)
				CSdev6 = dev5_to_m6(CSdev5, 0)
				dCSdev6 = CSdev6
				dCSdev6[:,1:,:] = dCSdev6[:,1:,:] - dCSdev6[:,:-1,:]

				# hydrostatic
				dHSvol = dHS_[...,:3]
				dCShyd6 = K*dHSvol.sum(dim=-1, keepdim=True)*norm
				
				# assemble
				dCS = dCSdev6 + dCShyd6
				J_pred.append(
					dCSdHS_to_DDSDDE(dCS, dHS, symmetrise=False, create_graph=True))		# (b, 1, 6, 6)
		
		if y_pred_.requires_grad:
			J_pred = torch.cat(J_pred, dim=1).flatten(-2,-1)								# (b, s, 36)
		
		y_pred = torch.cat(y_pred, dim=1)
		alphas = torch.stack(alphas, dim=1)
		return y_pred, J_pred, alphas

	def step_forward(self, data, criterion: dict) -> dict:
		y_pred, _, alphas, _ = self.forward(data.dHSdev5)

		loss = criterion.get('mae')(y_pred, data.CSdev5)
		mape = criterion.get('mape')(y_pred, data.CSdev5)
		acc = 1 - mape
		corr = criterion.get('corr')(dev5_to_dev6(y_pred), dev5_to_dev6(data.CSdev5))
		lossA = criterion.get('maeA')(alphas)
		# lossJ = criterion.get('mae')(J_pred, data.dCSdE[:,1:,:]) if len(J_pred)>0 else torch.zeros(1)

		return {
			'loss': loss,
			'mape': mape,
			'accuracy': acc if acc>=0 else torch.zeros(1),
			'corr11': corr[0], 'corr22': corr[1], 'corr33': corr[2], 
			'corr44': corr[3], 'corr55': corr[4], 'corr66': corr[5],  
			'lossA': lossA, 
			# 'lossJ': lossJ / 7.4972e+10, 		# 1/K, inverse of Bulk modulus scaling 
		}

	def step_backward(self, kwloss: dict, optimizers: dict) -> None:
		opt = optimizers.get('opt')
		loss = kwloss.get('loss')
		lossA = kwloss.get('lossA')
		# lossJ = kwloss.get('lossJ')

		lmbA = 10
		lmbJ = 1.5

		opt.zero_grad()
		# (loss + lmbA*lossA + lmbJ*lossJ).backward()
		(loss + lmbA*lossA).backward()
		# (loss + lmbJ*lossJ).backward()
		# loss.backward()
		opt.step()
