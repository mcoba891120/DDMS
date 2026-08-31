import torch
import damask
from functorch import vmap 
from damask.mechanics import stress_Cauchy
from torch import Tensor
from typing import Dict
from . import tensor as _tensor

def FeFp_to_F(Fe:Tensor, Fp:Tensor) -> Tensor:	
	"""
    Computes deformation gradient from Fe and Fp.

    Args:
        Fe: elastic deformation gradient. Fe.shape == (\*, 9) or (\*, 3, 3).
        Fp: plastic deformation gradient. Fp.shape == (\*, 9) or (\*, 3, 3).

    Returns:
        the deformation gradient.
    """
	return torch.einsum('...ij, ...jk -> ...ik', Fe, Fp)

def P_to_CS(P:Tensor, F:Tensor, MAP=None) -> Tensor:
	"""
    Convert first Piola-Kirchoff stress to Cauchy stress.

    Args:
        P: The first Piola-Kirchhoff stress tensor. P.shape == (\*, 9) or (\*, 3, 3).
        F: The deformation gradient tensor. F.shape == (\*, 9) or (\*, 3, 3).
        MAP: The tensor notation map.

    Returns:
        The Cauchy stress tensor.
    """
	assert P.size() == F.size(), f'get P.size(): {P.size()} and F.size(): {F.size()}'
	if P.size(-1) == 9:
		bz = P.size()[:-1]
		P = P.reshape(*bz, 3, 3)
		F = F.reshape(*bz, 3, 3)
	CS = torch.from_numpy(stress_Cauchy(P, F))
	return CS if MAP is None else _tensor.sym33_to_m6(CS, MAP)

def dPdF_to_dCSdE(dPdF:Tensor, P:Tensor, F:Tensor, MAP=None) -> Tensor:
	"""
	Convert dPdF to dCSdE

	Args:
		dPdF: The fourth order stiffness tensor. dPdF.shape == (\*, 81), fortran order 
		P: The first Piola-Kirchhoff stress tensor. P.shape == (\*, 9) or (\*, 3, 3)
		F: The deformation gradient tensor. F.shape == (\*, 9) or (\*, 3, 3)
		MAP: The tensor notation map.
	
	Returns:
		The stiffness tensor dCSdE.
	"""
	assert P.size() == F.size(), f'get P.size(): {P.size()} and F.size(): {F.size()}'

	if P.size(-1) == 9:
		bz = P.size()[:-1]
		P = P.reshape(*bz, 3, 3)								# (*bz, 3, 3)
		F = F.reshape(*bz, 3, 3)								# (*bz, 3, 3)
	else:
		bz = P.size()[:-2]
	Kirchhoff = torch.einsum('...ij, ...kj -> ...ik', P, F)		# (*bz, 3, 3)
	J_inv = 1 / torch.linalg.det(F).reshape(*bz,1,1,1,1)		# (*bz, 1, 1, 1, 1)
	dPdF = _tensor.m81_to_m3333(dPdF)							# (*bz, 3, 3, 3, 3)

	# modified from DAMASK `CPFEM.f90`, torch efficient version, calculation VERIFIED
	H = torch.einsum('...jm,...ln,...imkn -> ...ijkl', F, F, dPdF)
	H -= torch.einsum('...jl,...im,...km -> ...ijkl', torch.eye(3), F, P)*3
	H += 0.5*(torch.einsum('...jl,...ik -> ...ijkl', Kirchhoff, torch.eye(3))*9 + \
			  torch.einsum('...ik,...jl -> ...ijkl', Kirchhoff, torch.eye(3))*9 + \
			  torch.einsum('...jk,...il -> ...ijkl', Kirchhoff, torch.eye(3))*9 + \
			  torch.einsum('...il,...jk -> ...ijkl', Kirchhoff, torch.eye(3))*9)

	H_sym = 0.25*(H + torch.einsum('...ijkl -> ...jikl', H) + 
					  torch.einsum('...ijkl -> ...ijlk', H) + 
					  torch.einsum('...ijkl -> ...jilk', H))
	dCSdE = J_inv*H_sym
	return _tensor.sym3333_to_m66(dCSdE, MAP) if MAP is not None else dCSdE

def FCS_to_dev5(grads:Tensor, sigma:Tensor, dim:int) -> Dict[str, Tensor]:
	"""
	Convert `F-CS` data into `dHS-CS dev5` data, along time dimension `dim`

	Args:
		grads: The deformation gradient tensor. grads.shape == (\*, 3, 3)
		sigma: The Cauchy stress tensor. sigma.shape == (\*, 3, 3)
		dim: The dimension of time.

	Returns:
		The dictionary of converted tensor.
	"""
	t_idx = torch.arange(grads.size(dim))

	eps = torch.from_numpy(damask.mechanics.strain(grads, t='U', m=0))		# Hencky strain, i.e., log strain
	deps = eps.index_select(dim, t_idx[1:]) - \
			eps.index_select(dim, t_idx[:-1])								# (*, s-1, *, 3, 3)
	
	deps_hyd, deps_dev = _tensor.m33_to_HydDev(deps)
	dHS 	= _tensor.sym33_to_m6(deps)
	dHShyd6 = _tensor.sym33_to_m6(deps_hyd)

	sigma_hyd, sigma_dev = _tensor.m33_to_HydDev(sigma)
	CS 		= _tensor.sym33_to_m6(sigma)
	CShyd6 	= _tensor.sym33_to_m6(sigma_hyd)

	# align time with deps_dev
	sigma_dev_ = sigma_dev.index_select(dim, t_idx[1:])						# (*, s-1, *, 3, 3)
	CS = CS.index_select(dim, t_idx[1:])									# (*, s-1, *, 6)
	CShyd6 = CShyd6.index_select(dim, t_idx[1:])							# (*, s-1, *, 3, 3)

	# network input / output
	dHSdev5 = _tensor.m33_to_dev5(deps_dev)
	CSdev5 = _tensor.m33_to_dev5(sigma_dev_)

	return {
		'eps': eps,						# (*, s, *, 3, 3)
		'dHS': dHS,						# (*, s-1, *, 6)
		'dHSdev5': dHSdev5,				# (*, s-1, *, 5)
		'dHShyd6': dHShyd6, 			# (*, s-1, *, 6)
		'sigma': sigma,					# (*, s, *, 3, 3)
		'CS': CS, 						# (*, s-1, *, 6)
		'CSdev5': CSdev5, 				# (*, s-1, *, 5)
		'CShyd6': CShyd6, 				# (*, s-1, *, 6)
	}

def UCS_to_dev5(U_EXP6:Tensor, CS_IMP6:Tensor) -> Dict[str, Tensor]:
	"""
	Convert `U-CS` data into `dHS-CS dev5` data, where `U` follows EXP notation, `CS` follows IMP notation,

	Args:
		U_EXP6: The stretch tensor. U_EXP6.shape == (\*, 6)
		CS_IMP6: The Cauchy stress tensor. CS_IMP6.shape == (\*, 6)

	WARNING: 
		1. assuming F == U
		2. assuming second dimension represents time 
	"""
	grads = _tensor.m6_to_sym33(U_EXP6, MAP=_tensor.MAP.EXP6tom33)
	sigma = _tensor.m6_to_sym33(CS_IMP6)
	return FCS_to_dev5(grads, sigma, dim=1)

def dHS6_to_CShyd6(dHS6:Tensor, K=7.4972e+10, dim:int=0) -> Tensor:
	"""
	Calculate hydrostatic stress from Hencky Strain increment and Bulk modulus along time dimension `dim`
	
	Args:
		dHS6: The Hencky strain increment. dHS6.shape == (\* , seq_len, \*, 6), should follow EXP or IMP notation
		K: The Bulk modulus
		dim: The dimension of time.
	"""
	CSmean = 0
	CShyd6 = torch.zeros_like(dHS6)
	for t_idx in torch.arange(dHS6.size(dim)):
		dHSVol = dHS6.index_select(dim, t_idx)[...,:3].sum()
		CSmean += K * dHSVol
		CShyd6.index_add_(dim, t_idx, CSmean*torch.tensor([[1,1,1,0,0,0]]))
	return CShyd6

def dCSdHS_to_DDSDDE(dCS:Tensor, dHS:Tensor, symmetrise=True, **grad_kwargs) -> Tensor:
	"""
	Calculate consistent stiffness matrix (DDSDDE) from Cauchy stress increment and Hencky strain increment, using pytorch `grad` + `vmap`

	Args:
		dCS: The Cauchy stress increment. dCS.size() == (*, 6)
		dHS: The Hencky strain increment. dHS.size() == (*, 6)
		symmetrise: Whether to symmetrise the result stiffness or not.
		grad_kwargs: arguments to pass into `vmap`

	Returns:
		The stiffness matrix (DDSDDE)
	"""
	assert dCS.requires_grad == dHS.requires_grad == True, 'dCS or dHS not having gradient !'

	I = torch.eye(dCS.size(-1), device=dCS.device).repeat(*dCS.size()[:-1],1,1)
	jac = vmap(lambda y, x, v: torch.autograd.grad(y, x, v, retain_graph=True, **grad_kwargs)[0], in_dims=(None, None, -2), out_dims=-2)
	J = jac(dCS, dHS, I)
	if symmetrise:
		J = (J + torch.transpose(J, -2, -1)).div(2)
	return J
