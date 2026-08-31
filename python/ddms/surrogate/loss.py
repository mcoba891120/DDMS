import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np

class Correlation(Module):
	""" Pearson correlation """
	def __init__(self):
		super().__init__()

	def forward(self, y_preds, ys, return_avg=True):
		"""
        Calculates the correlation between the predicted and ground truth values.

        Args:
            y_preds: The predicted values.
            ys: The ground truth values.
            return_avg: Whether to return the average correlation coefficient or a list of correlation coefficients.

        Returns:
            The correlation coefficient or a list of correlation coefficients.
        """
		v_dim = y_preds.size(-1)

		Rs = []
		for y_pred, y in zip(y_preds.detach().cpu().numpy(), ys.detach().cpu().numpy()):
			r = np.corrcoef(y_pred, y, rowvar=False)	# (2*v_dim, 2*v_dim)
			r = r[:v_dim, v_dim:].diagonal()			# (v_dim,)
			if not np.any(np.isnan(r)):
				Rs.append(r)
		Rs = torch.from_numpy(np.array(Rs))
		
		if return_avg:
			return Rs.mean(axis=0)
		return Rs

class MAEAlpha(Module):
	"""
	see https://doi.org/10.1016/j.ijplas.2022.103430
	"""
	def __init__(self, tilda_alpha=500):
		super().__init__()
		self.tilda_alpha = tilda_alpha

	def forward(self, alpha):
		"""
        Calculates the mean absolute error of alpha.

        Args:
            alpha: The alpha values.

        Returns:
            The mean absolute error.
        """
		lossA = torch.relu(torch.log(alpha/self.tilda_alpha))
		lossA = lossA.nan_to_num().mean()
		return lossA

class MAEJ(Module):
	"""
	see https://doi.org/10.1016/j.ijplas.2022.103430
	"""
	def __init__(self):
		super().__init__()

	def forward(self, y_pred, y, x):
		"""
        Calculates the mean absolute error of the Jacobian.

        Args:
            y_pred: The predicted values.
            y: The ground truth values.
            x: The input values.

        Returns:
            The mean absolute error of the Jacobian.
        """
		# add zero
		b, _, c = y_pred.size()
		y_pred 	= torch.cat([y_pred.new_zeros((b,1,c)), y_pred], dim=1)
		y 		= torch.cat([y.new_zeros((b,1,c)), y], dim=1)

		dy_pred = y_pred[:,1:,:] - y_pred[:,:-1,:]
		dy 		= y[:,1:,:] - y[:,:-1,:]

		lossJ = torch.linalg.norm(dy_pred-dy, dim=-1) / torch.linalg.norm(x, dim=-1)
		lossJ = lossJ.nan_to_num().mean()
		return lossJ

class MARE(Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_pred, y):
		"""
        Calculates the mean absolute relative error.
	
		WARNING: divided by zero
			y should not be zero

        Args:
            y_pred: The predicted values.
            y: The ground truth values.

        Returns:
            The mean absolute relative error.
        """
		# (seq_len, batch_num_grain, 10)
		loss = torch.abs(y-y_pred)/y
		return loss.mean()

class AMRPD(Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_pred, y):
		"""
        Calculates the absolute mean relative percentage difference.

        Args:
            y_pred: The predicted values.
            y: The ground truth values.

        Returns:
            The absolute mean relative percentage difference.
        """
		# (seq_len, batch_num_grain, 10)
		loss = 2*(y-y_pred)/(y.abs()+y_pred.abs())
		return loss.abs().mean()

class MAPE(Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_pred, y):
		"""
        Calculates the mean absolute percentage error.

        Args:
            y_pred: The predicted values.
            y: The ground truth values.

        Returns:
            The mean absolute percentage error.
        """
		ape = ((y_pred-y).abs()/(y.abs()+1e-8)).cpu().detach()			# pytorch-focasting implementation
		q25, q75 = torch.quantile(ape, torch.tensor([0.25, 0.75]))
		lower, upper = q25-1.5*(q75-q25), q75+1.5*(q75-q25)
		mask = torch.logical_and(ape>lower, ape<upper)
		mape = ape[mask].mean()
		return mape

class Accuracy(Module):
	def __init__(self):
		super().__init__()
		self.mape = MAPE()

	def forward(self, y_pred, y):
		acc = 1 - self.mape(y_pred, y)
		return max(acc, 0)

class PINN(Module):
	"""
	WARNING: 
		deprecated
	"""
	def __init__(self):
		super().__init__()
		self.mae = torch.nn.L1Loss()
		# self.mae = torch.nn.MSELoss()
	
	def getW(self, sigma, F, volume):
		J = torch.det(F)                                                    # (batch_num_grain, seq_len)
		P = torch.einsum('gs,gsij,gsji->gsij', J, sigma, F.inverse())       # (batch_num_grain, seq_len, 3, 3)
		delta_F = F[:,1:]-F[:,:-1]                                          # (batch_num_grain, seq_len-1, 3, 3)
		W_g = torch.einsum('gsij,gsij->g', P[:,1:], delta_F)
		return torch.dot(W_g, volume)/torch.sum(volume)

	def getR(self, F, volume, edge_index, normal_vec):
		permu = torch.zeros((3,3,3), device=F.device)
		permu[0,1,2] = permu[1,2,0] = permu[2,0,1] = 1
		permu[2,1,0] = permu[1,0,2] = permu[0,2,1] = -1

		g1_idx, g2_idx = torch.unique(edge_index, dim=1)
		G_g_alpha = (F[g2_idx]-F[g1_idx])/2                                         			# (batch_num_edges, seq_len, 3, 3)
		# G_g_alpha.clamp_(min=1e-5) 															# TODO: prevent inf gradient, TODO: keep getting NAN
		N_g_alpha = torch.einsum('ek,esil,jkl->esij', \
								-normal_vec[g1_idx], G_g_alpha, permu)                   		# (batch_num_edges, seq_len, 3, 3)
		R = torch.sinh(torch.norm(N_g_alpha, dim=(2,3)))*volume[g1_idx]                         # (batch_num_edges, seq_len)
		R = R/torch.sum(volume)                                                                 # (batch_num_edges, seq_len)
		return R
	
	def forward(self, O_pred=None, O=None, S_pred=None,S=None,F_pred=None,F=None,data=None,volume=None):
		"""
		Args:
			data:
				x_t: 			(batch_num_nodes, seq_len+1, 20), 	props(2), eulers(3), sigma(6), grads(9)
				eps:  			(batch_num_nodes, seq_len, 6),		strain(6)
				edge_index: 	(2, batch_num_edges) 				
				edge_attr: 		(batch_num_edges, 5) 				n(3), misorientaion(1), surfaceArea(1)
			L1: MAE of orientation
			L2: MAE of Cauchy stress
			L3: MAE of deformation gradient
			L4: work energy density 
			L5: deformation gradient mismatch
		"""
		if data is not None:
			volume = data.x_t[:,0,1:2] if volume is None else volume                # (batch_num_node, 1)
			edge_index = data.edge_index 											# (2, batch_num_edges)
			normal_vec = data.edge_attr[:,:3] 										# (batch_num_edges, 3)

		L1 = L2 = L3 = L4 = L5 = W_pred = W_sim = R_pred = R_sim = \
			torch.zeros(1, requires_grad=True, device=volume.device)

		# >----------------------------------------------------------------------------------------------------
		if O_pred is not None and O is not None:
			L1 = self.mae(O_pred, O)

		# >----------------------------------------------------------------------------------------------------
		if S_pred is not None and S is not None:
			S_pred = S_pred[...,[[0,5,4],[5,1,3],[4,3,2]]]              			# sym 6 -> 9, (batch_num_grain, seq_len, 3, 3)
			S = S[...,[[0,5,4],[5,1,3],[4,3,2]]]                        			# sym 6 -> 9, (batch_num_grain, seq_len, 3, 3)
			L2 = self.mae(S_pred, S)

		# >----------------------------------------------------------------------------------------------------
		if F_pred is not None and F is not None:
			F_pred = F_pred[...,[[0,1,2],[3,4,5],[6,7,8]]]                  		# (batch_num_grain, seq_len, 3, 3)
			F = F[...,[[0,1,2],[3,4,5],[6,7,8]]]                            		# (batch_num_grain, seq_len, 3, 3)
			L3 = self.mae(F_pred, F)

		# >---------------------------------------------------------------------------------------------------- 
		# # TODO: cannot inverse
		# W_pred = self.getW(sigma_pred, F_pred, volume)
		# W_sim = self.getW(sigma, F, volume)
		# L4 = self.mae(W_pred-W_sim)

		# >----------------------------------------------------------------------------------------------------
		if F_pred is not None and F is not None and data is not None:
			# R_pred = self.getR(F_pred, volume, edge_index, normal_vec)               # (batch_num_edges, seq_len)
			# R_sim = self.getR(F, volume, edge_index, normal_vec)                     # (batch_num_edges, seq_len)
			# L5 = self.mae(R_pred, R_sim)
			...

		return L1, L2, L3, L4, L5, W_pred, W_sim, R_pred.mean(), R_sim.mean()

class GeneratorLoss(nn.Module):
	"""
	WARNING: 
		deprecated
	"""
	def __init__(self):
		"""
		Cross entropy loss implemetation of 
		"Deep learning model to predict complex stress 
		and strain fields in hierarchical composites"
		"""
		super().__init__()
		self.L1Loss = nn.L1Loss()
		# self.L1Loss = nn.L1Loss()
	
	def forward(self, y_gen, y_real, dis_gen):
		"""
		Parameters:
		---------
		y_gen: generator predictions
		y_real: real data
		dis_gen: discrimination of y_gen
		"""
		loss_gen = self.L1Loss(y_gen, y_real)
		loss_gan = self.L1Loss(dis_gen, torch.ones_like(dis_gen))
		return loss_gen, loss_gan

class DiscriminatorLoss(nn.Module):
	"""
	WARNING: 
		deprecated
	"""
	def __init__(self):
		"""
		Cross entropy loss implemetation of 
		"Deep learning model to predict complex stress 
		and strain fields in hierarchical composites"
		"""
		super().__init__()
		self.L1Loss = nn.L1Loss()
	
	def forward(self, dis_gen, dis_real):
		"""
		Parameters:
		---------
		dis_gen: discrimination of y_gen
		dis_real: discrimination of y_real
		"""
		loss_real = self.L1Loss(dis_real, torch.ones_like(dis_real))
		loss_fake = self.L1Loss(dis_gen, torch.zeros_like(dis_gen))
		return loss_real, loss_fake
	

	