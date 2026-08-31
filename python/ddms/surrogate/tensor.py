import torch
from torch import Tensor
from typing import Tuple

class MAP:
	"""
	The tensor notation map, including Nye(NYE), abaqus implicit(IMP) and abaqus explicit(EXP).

	Notation: Tensor index of each notation
		```
		NYE 11 22 33 12 23 13
		IMP 11 22 33 12 13 23
		EXP 11 22 33 12 23 13
		```

	Attributes:
		m3333toEXP66: The mapping from 3x3x3 to 6x6 following the explicit notation.
		m3333toIMP66: The mapping from 3x3x3 to 6x6 following the implicit notation.
		m33toEXP6: The mapping from 3x3 to 6 following the explicit notation.
		m33toIMP6: The mapping from 3x3 to 6 following the implicit notation.
		EXP6tom33: The mapping from 6 to 3x3 following the explicit notation.
		IMP6tom33: The mapping from 6 to 3x3 following the implicit notation.
	"""
	m3333toEXP66 = torch.tensor([[0,1,2,0,1,0], 
				 				 [0,1,2,1,2,2]])
	m3333toIMP66 = torch.tensor([[0,1,2,0,0,1], 
				 				 [0,1,2,1,2,2]])
	m33toEXP6 = torch.tensor([0,4,8,1,5,2])
	m33toIMP6 = torch.tensor([0,4,8,1,2,5])
	EXP6tom33 = torch.tensor([0,3,5,
							  3,1,4,
							  5,4,2])
	IMP6tom33 = torch.tensor([0,3,4,
							  3,1,5,
							  4,5,2])
	
def delta(i, j) -> int:
	"""
	Kronecker delta.

	Args:
		i: The first index.
		j: The second index.

	Returns:
		1 if i == j, 0 otherwise.
	"""
	return 1 if i==j else 0

def polar(m33:Tensor) -> Tensor:
	"""
    Right side polar decomposition.

    Args:
        m33: The 3x3 tensor. m33.shape == (\*, 3, 3)

    Returns:
        The tensor u and p, where u is the rotation matrix and p is the stretch tensor.
    """
	w, s, vh = torch.linalg.svd(m33, False)
	u = torch.matmul(w, vh)
	p = torch.matmul(torch.einsum('...ij, ...j -> ...ij', vh.transpose(-2,-1).conj(), s), vh)
	return u, p

def m81_to_m3333(m81:Tensor, order:str='F') -> Tensor:
	"""
    Expand 81 vector to 3x3x3x3 tensor following `order`(default=fortran).

    Args:
        m81: The 81-dimensional vector. m81.shape == (\*, 81).
        order: The order of the output tensor. 'F' for fortran, 'C' for C++.

    Returns:
        The 3x3x3x3 tensor.
    """
	bz = m81.size()[:-1]
	m3333 = m81.reshape(*bz,3,3,3,3)
	if order=='F':
		return torch.einsum('...ijkl->...lkji', m3333)
	return m3333

def sym33_to_m6(m33:Tensor, MAP=MAP.m33toIMP6) -> Tensor:
	"""
    Reduce symmetric 3x3 tensor to 6 following `MAP(default=IMP)` notation.

    Args:
        m33: The 3x3 tensor. m33.shape == (\*, 3, 3).
        MAP: The tensor notation map.

    Returns:
        The 6-dimensional tensor.
    """
	return m33.flatten(-2, -1)[..., MAP]

def m6_to_sym33(m6:Tensor, MAP=MAP.IMP6tom33) -> Tensor:
	"""
    Expand 6 tensor to symmetric 3x3 following `MAP(default=IMP)` notation.

    Args:
        m6: The 6-dimensional tensor. m6.shape == (\*, 6).
        MAP: The tensor notation map.

    Returns:
        The 3x3 tensor.
    """
	return m6[..., MAP].reshape(*m6.size()[:-1], 3, 3)

def sym3333_to_m66(m3333:Tensor, MAP=MAP.m3333toIMP66) -> Tensor:
	"""
    Reduce symmetric 3x3x3x3 tensor to 6x6 tensor following `MAP(default=IMP)` notation.

    Args:
        m3333: The 3x3x3x3 tensor. m3333.shape == (\*, 3, 3, 3, 3).
        MAP: The tensor notation map.

    Returns:
        The 6x6 tensor.
    """
	bz = m3333.size()[:-4]
	m66 = m3333.new_zeros((*bz,6,6))
	for i in range(6):
		for j in range(6):
			m66[...,i,j] = m3333[..., MAP[0,i], MAP[1,i], MAP[0,j], MAP[1,j]]
	return m66

def m33_to_dev5(m33:Tensor) -> Tensor:
	"""
	Convert 3x3 tensor into 5 dimensional deviatoric tensor following https://doi.org/10.1016/j.ijplas.2022.103430.
	Notes:
		The result is equivalent with `m33` or `dev33` being input
	
	Args:
		m33: The 3x3 tensor. m33.shape == (\*, 3, 3)

	Returns:
		The 5 dimensional deviatoric tensor.
	"""
	dev5 = m33.new_zeros((*m33.size()[:-2], 5))
	dev5[...,0] = (m33[...,0,0] - m33[...,1,1]).div(torch.sqrt(torch.tensor(2)))
	dev5[...,1] = (2*m33[...,2,2] - m33[...,0,0] - m33[...,1,1]).div(torch.sqrt(torch.tensor(6)))
	dev5[...,2] = torch.sqrt(torch.tensor(2))*m33[...,0,1]
	dev5[...,3] = torch.sqrt(torch.tensor(2))*m33[...,0,2]
	dev5[...,4] = torch.sqrt(torch.tensor(2))*m33[...,1,2]
	return dev5

def m6_to_dev5(m6:Tensor, MAP=MAP.IMP6tom33) -> Tensor:
	"""
    Convert 6-dimensional tensor to 5-dimensional deviatoric tensor following the specified mapping.

    Args:
        m6: The 6-dimensional tensor. m6.shape == (\*, 6)
        MAP: The tensor notation map.

    Returns:
        The 5-dimensional deviatoric tensor.
    """
	return m33_to_dev5(m6_to_sym33(m6, MAP))

def dev5_to_dev6(dev5:Tensor) -> Tensor:
	"""
    Convert 5-dimensional deviatoric tensor to 6 following implicit notation.

    Args:
        dev5: The 5-dimensional deviatoric tensor. dev5.shape == (\*, 5).

    Returns:
        The 6-dimensional deviatoric tensor.
    """
	dev6 = dev5.new_zeros((*dev5.size()[:-1], 6))

	zz = dev5[...,1]*torch.sqrt(torch.tensor(6))/3		# tr(dev) = 0
	xx = (dev5[...,0]*torch.sqrt(torch.tensor(2)) - 
				(dev5[...,1]*torch.sqrt(torch.tensor(6))-2*zz))/2
	yy = xx - dev5[...,0]*torch.sqrt(torch.tensor(2))
	xy = dev5[...,2] / torch.sqrt(torch.tensor(2))
	xz = dev5[...,3] / torch.sqrt(torch.tensor(2))
	yz = dev5[...,4] / torch.sqrt(torch.tensor(2))

	dev6[...,0] = xx; dev6[...,1] = yy; dev6[...,2] = zz
	dev6[...,3] = xy; dev6[...,4] = xz; dev6[...,5] = yz
	return dev6

def dev5_to_m6(dev5:Tensor, hyd6:Tensor) -> Tensor:
	"""
    Convert deviatoric tensor 5 to tensor 6 (implicit notation)

    Args:
        dev5: The 5-dimensional deviatoric tensor. dev5.shape == (\*, 5).
        hyd6: The 6-dimensional hydrostatic tensor. hyd6.shape == (\*, 6).

    Returns:
        The 6-dimensional tensor.
    """
	dev6 = dev5_to_dev6(dev5)
	return dev6 + hyd6

def m33_to_HydDev(m33:Tensor) -> Tuple[Tensor, Tensor]:
	"""
    Decompose the matrix `m33` into hydrostatic / deviatoric part.

    Args:
        m33: The 3x3 tensor. m33.shape == (\*, 3, 3).

    Returns:
        A tuple of the hydrostatic tensor and the deviatoric tensor.
    """
	hyd = torch.einsum('...ii, jk -> ...jk', m33, torch.eye(3, device=m33.device)).div(3)
	dev = m33 - hyd
	return hyd, dev

def m6_to_HydDev(m6:Tensor, MAP=MAP.IMP6tom33) -> Tuple[Tensor, Tensor]:
	"""
    Decompose the tensor `m6` into hydrostatic / deviatoric part.

    Args:
        m6: The 6-dimensional tensor.
        MAP: The tensor notation map.

    Returns:
        A tuple of the hydrostatic tensor and the deviatoric tensor.
    """
	return m33_to_HydDev(m6_to_sym33(m6, MAP))



