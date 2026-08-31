import numpy as np
import matplotlib.pyplot as plt 
from torch import Tensor
from numpy import ndarray
from matplotlib.figure import Figure
from damask.mechanics import (
	equivalent_strain_Mises, equivalent_stress_Mises)
from typing import List, Tuple


class VIZ():
	"""
	The title of visualized plot.

	Notation: Tensor index of each notation
		```
		NYE 11 22 33 12 23 13
		IMP 11 22 33 12 13 23
		EXP 11 22 33 12 23 13
		```

	Attributes:
		names_IMP6: The titles of the visualized plots for the IMP notation.
		names_EXP6: The titles of the visualized plots for the EXP notation.
		names_m33: The titles of the visualized plots for the 3x3 notation.
	"""
	names_IMP6 = [['$\\sigma_{11}$[MPa]', '$\\sigma_{12}$[MPa]'],
				  ['$\\sigma_{22}$[MPa]', '$\\sigma_{13}$[MPa]'],
				  ['$\\sigma_{33}$[MPa]', '$\\sigma_{23}$[MPa]']]
	names_EXP6 = [['$\\sigma_{11}$[MPa]', '$\\sigma_{12}$[MPa]'],
				  ['$\\sigma_{22}$[MPa]', '$\\sigma_{23}$[MPa]'],
				  ['$\\sigma_{33}$[MPa]', '$\\sigma_{13}$[MPa]']]
	names_m33 = [['$\\sigma_{11}$[MPa]', '$\\sigma_{12}$[MPa]', '$\\sigma_{13}$[MPa]'],
				 ['$\\sigma_{21}$[MPa]', '$\\sigma_{22}$[MPa]', '$\\sigma_{23}$[MPa]'],
				 ['$\\sigma_{31}$[MPa]', '$\\sigma_{32}$[MPa]', '$\\sigma_{33}$[MPa]']]

def find_most(y_pred:Tensor, y:Tensor, prev=None, return_index=False) -> List[List[Tensor]]:
	"""
    Find the best / worst prediction in whole dataset.

    Args:
        y_pred: Batch of prediction of model
        y: Batch of ground truth
        prev: History min max results
		return_index: Whether to return min/max index or not.

    Returns:
		tuple((min_loss, min_y_pred, min_y, i_min), (max_loss, max_y_pred, max_y, i_max)) if return_index is `True`, 
		tuple((min_loss, min_y_pred, min_y), (max_loss, max_y_pred, max_y)) otherwise.
        
    """
	assert y.ndim==y_pred.ndim==3, f'y is {y.ndim} ndim and y_pred is {y_pred.ndim} ndim'		# (num_node, seq_len, features)
	y_pred = y_pred.detach().cpu()
	y = y.detach().cpu()

	sample_loss = (y_pred-y).abs().flatten(-2,-1).mean(dim=1)									# (b, )
	i_min, min_loss = sample_loss.argmin(), sample_loss.min()
	i_max, max_loss = sample_loss.argmax(), sample_loss.max()
	min_y_pred, min_y  = y_pred[i_min], y[i_min]
	max_y_pred, max_y  = y_pred[i_max], y[i_max]

	if return_index:
		curr = [[min_loss, min_y_pred, min_y, i_min], [max_loss, max_y_pred, max_y, i_max]]
	else:
		curr = [[min_loss, min_y_pred, min_y], [max_loss, max_y_pred, max_y]]
		
	if prev is not None:
		if curr[0][0]>prev[0][0]:
			curr[0] = prev[0]
		if curr[1][0]<prev[1][0]:
			curr[1] = prev[1]
	return curr

def get_homogenized_xy(y:ndarray, fill_first=True) -> Tuple[ndarray, ndarray]:
	"""
    Get homogenized xy array.

    Args:
        y: The 3D array of the data, each dimension should be (batch, sequence, channels)
        fill_first: Whether to fill the first entry of the xy array with 0 or not.

    Returns:
        A tuple of the x and y arrays.
    """
	assert y.ndim==3 												# (batch_num_node, seq_len, channels)
	seq_len, feature_dim = y.shape[-2:]
	step = np.linspace(0, seq_len, seq_len+1).reshape(-1,1) 		# (seq_len+1, 1)
	features = np.zeros((seq_len+1, feature_dim))
	features[1:] = y.mean(axis=0)
	if fill_first:
		return step, features
	return step[:-1], features[1:]

def plot_homogenized(y:Tensor, y_pred:Tensor, names:List[str], allInOne:bool=False, **kwargs) -> Figure:
	""" 
	Plot time vs. homogenized y/y_pred curve.
	
	Args:
		y: The ground truth, each dimension should be (batch, sequence, channels)
		y_pred: The model prediction, each dimension should be (batch, sequence, channels).
		names: titles of plot if `allInOne` is False, otherwise labels of plot, 
				in both cases its shape will be considered as subplots shape
		allInOne: Whether to plot all curve in one figure.
		kwargs: 
			- order: The plot order, should be 'F' or 'C'.
			- shift: The tensor to shift y/y_pred.
			- fill_first: See `get_homogenized_xy`.
			- figsize: The matplotlib `figsize`.
			- legend: Whether to show legend or not.
			- twinx: Broken, don't use.

	Returns:
		The matplotlib figure.
	"""
	assert y.ndim==y_pred.ndim==3, f'y is {y.ndim} ndim and y_pred is {y_pred.ndim} ndim'		# (num_node, seq_len, features)
	
	if isinstance(y, Tensor):
		y = y.detach().cpu().numpy()
	if isinstance(y_pred, Tensor):
		y_pred = y_pred.detach().cpu().numpy()

	order 			= kwargs.get('order', 'C')
	shift 			= kwargs.get('shift', None) 				# shift normal component of grads
	fill_first 		= kwargs.get('fill_first', True)
	figsize 		= kwargs.get('figsize', None)
	legend			= kwargs.get('legend', True)
	twinx			= kwargs.get('twinx', 0)
	sim_x, sim_y 	= get_homogenized_xy(y, fill_first) 		# (seq_len+1, features)
	sur_x, sur_y 	= get_homogenized_xy(y_pred, fill_first) 	# (seq_len+1, features)
	names 			= np.array(names)

	if shift is not None:
		sim_y[0] += shift.flatten(order)
		sur_y[0] += shift.flatten(order)

	if allInOne:
		fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
		ax.set_xlabel('increment')
		ax.set_ylabel('$\\sigma_{ii}$[MPa]')
		if twinx:
			ax2 = ax.twinx()
			ax2.set_ylabel('$\\sigma_{ij}$[MPa]')

		for i, label in enumerate(names.flatten()):
			if twinx and i >= twinx:
				ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
				lines = ax2.plot(sim_x, sim_y[:,i], '-', alpha=0.5, label=f'{label}', c='blue')
				ax2.plot(sur_x, sur_y[:,i], '--o', c=lines[0].get_color(), alpha=0.5, markersize=3)
			else:
				lines = ax.plot(sim_x, sim_y[:,i], '-', alpha=0.5, label=f'{label}', c='red')
				ax.plot(sur_x, sur_y[:,i], '--o', c=lines[0].get_color(), alpha=0.5, markersize=3)
	else:
		fig, axes = plt.subplots(*names.shape, figsize=figsize, tight_layout=True)
		if isinstance(axes, plt.Axes):
			axes = np.array([axes])
		for i, (ax, name) in enumerate(zip(axes.flatten(order), names.flatten(order))):
			ax.plot(sim_x, sim_y[:,i], label='simulation', c='cyan', alpha=0.8)
			ax.plot(sur_x, sur_y[:,i], label='surrogate', c='salmon', alpha=0.8)
			ax.set_xlabel('increment')
			ax.set_ylabel(name)
	if legend:
		plt.legend(loc=(1.1, 0))
	return fig

def plot_homogenized_vM(eps:Tensor, eps_pred, CS:Tensor, CS_pred:Tensor, **kwargs) -> Figure:
	""" 
	Plot homogenized Von Mises stress strain curve .

	Args:
		eps: Hencky strain(log strain). eps.shape == (\*, 3, 3).
		eps_pred: Hencky strain(log strain) prediction. eps_pred.shape == (\*, 3, 3).
		CS: Cauchy stress. CS.shape == (\*, 3, 3).
		CS_pred: Cauchy stress prediction. CS_pred.shape == (\*, 3, 3).
		kwargs: 
			- 'figsize': The matplotlib `figsize`.

	Returns:
		The matplotlib figure.
	"""
	assert eps.ndim==eps_pred.ndim==CS.ndim==CS_pred.ndim==3 						# (seq_len, 3, 3)
	
	if isinstance(eps, Tensor):
		eps = eps.cpu().numpy()
	if isinstance(eps_pred, Tensor):
		eps_pred = eps_pred.cpu().numpy()
	if isinstance(CS, Tensor):
		CS = CS.cpu().numpy()
	if isinstance(CS_pred, Tensor):
		CS_pred = CS_pred.cpu().numpy()

	figsize = kwargs.get('figsize', None)

	eps_vm = equivalent_strain_Mises(eps)
	eps_pred_vm = equivalent_strain_Mises(eps_pred)
	sigma_vm = equivalent_stress_Mises(CS)
	sigma_pred_vm = equivalent_stress_Mises(CS_pred)

	fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
	ax.plot(eps_vm, sigma_vm, c='cyan', label='simulation', alpha=0.8)
	ax.plot(eps_pred_vm, sigma_pred_vm, c='salmon', label='surrogate', alpha=0.8)
	ax.set_xlabel('Von Mises strain')
	ax.set_ylabel('Von Mises stress')
	ax.legend(loc=(1.1, 0))
	return fig


# >----------------------------------------------------------------------------------------------------
# DEPRECATED
# >----------------------------------------------------------------------------------------------------
# def plot_homogenized_v2(y:ndarray, y_pred:ndarray, names:List[str], allInOne:bool=False, **kwargs) -> Figure:
# 	""" 
# 	plot homogenized y vs. y_pred curve
	
# 	Args:
# 		y: 			ground truth 
# 		y_pred:		model prediction 
# 		names:		titles of plot if allInOne is False, otherwise labels of plot, 
# 					in both cases its shape will be considered as subplots shape
# 		allInOne: 	all in one plot
# 		kwargs:		can be 'order', 'shift', 'figsize'
# 	"""
# 	assert y.ndim==y_pred.ndim==3  								# (num_node, seq_len, features)
	
# 	if isinstance(y, Tensor):
# 		y = y.cpu().numpy()
# 	if isinstance(y_pred, Tensor):
# 		y_pred = y_pred.cpu().numpy()

# 	order 			= kwargs.get('order', 'C')
# 	shift 			= kwargs.get('shift', None) 				# shift normal component of grads
# 	figsize 		= kwargs.get('figsize', None)
# 	fill_first 		= kwargs.get('fill_first', False)
# 	sim_x, sim_y 	= get_homogenized_xy(y, fill_first) 		# (seq_len+1, features)
# 	sur_x, sur_y 	= get_homogenized_xy(y_pred, fill_first) 	# (seq_len+1, features)
# 	names 			= np.array(names)

# 	# # NNabaqus 2 scale 
# 	# df = pd.read_csv(f'{path}/StressStrain_NNabaqus.txt', sep='\s+')
# 	# FE2_y = df.iloc[:,1].values*1e6
# 	# _, FE2_y = get_homogenized_xy(FE2_y.reshape(1,-1,1), fill_first)
# 	# FE2_x = np.linspace(0, 100, len(FE2_y))

# 	if shift is not None:
# 		sim_y[0] += shift.flatten(order)
# 		sur_y[0] += shift.flatten(order)

# 	fig, axes = plt.subplots(*names.shape, figsize=figsize, tight_layout=True)
# 	if allInOne:
# 		for i, label in enumerate(names.flatten()):
# 			lines = axes.plot(sim_x, sim_y[:,i], '--', alpha=0.8)
# 			axes.plot(sur_x, sur_y[:,i], c=lines[0].get_color(), label=f'{label}', alpha=0.8)
# 	else:
# 		if isinstance(axes, plt.Axes):
# 			axes = np.array([axes])
# 		for i, (ax, name) in enumerate(zip(axes.flatten(order), names.flatten(order))):
# 			# ax.plot(sim_x, sim_y[:,i], label='simulation', c='cyan', alpha=0.6)
# 			# ax.plot(sur_x, sur_y[:,i], label='surrogate', c='salmon', alpha=0.6)
# 			# TODO
# 			# ax.plot(sur_x, kwargs.get('iter_CS_pred')[:,i], label='iter_CS_pred', c='green', alpha=0.6)
# 			ax.plot(kwargs.get('refine_x')[:-10], kwargs.get('refine_CS_pred')[:,i][:-10], label='Pytorch', color='blue', alpha=0.6)
# 			ax.scatter(kwargs.get('FE2_x')[::10][:-1], kwargs.get('FE2_y')[::10,i][:-1], label='Abaqus+Pytorch', c='black', alpha=0.6, facecolors='none', s=10)
# 			# TODO
# 			ax.set_xlabel('loadstep')
# 			ax.set_ylabel(f'{name}(MPa)')
# 	plt.legend(loc=(1.1, 0))
# 	return fig

# def plot_homogenized_v3(y:ndarray, y_pred:ndarray, names:List[str], allInOne:bool=False, **kwargs) -> Figure:
# 	""" 
# 	plot homogenized y vs. y_pred curve
	
# 	Args:
# 		y: 			ground truth 
# 		y_pred:		model prediction 
# 		names:		titles of plot if allInOne is False, otherwise labels of plot, 
# 					in both cases its shape will be considered as subplots shape
# 		allInOne: 	all in one plot
# 		kwargs:		can be 'order', 'shift', 'figsize'
# 	"""
# 	assert y.ndim==y_pred.ndim==3  								# (num_node, seq_len, features)
	
# 	if isinstance(y, Tensor):
# 		y = y.cpu().numpy()
# 	if isinstance(y_pred, Tensor):
# 		y_pred = y_pred.cpu().numpy()

# 	order 			= kwargs.get('order', 'C')
# 	shift 			= kwargs.get('shift', None) 				# shift normal component of grads
# 	figsize 		= kwargs.get('figsize', None)
# 	fill_first 		= kwargs.get('fill_first', True)
# 	sim_x, sim_y 	= get_homogenized_xy(y, fill_first) 		# (seq_len+1, features)
# 	sur_x, sur_y 	= get_homogenized_xy(y_pred, fill_first) 	# (seq_len+1, features)
# 	names 			= np.array(names)

# 	# surrogate / damask
# 	F = kwargs.get('F', None) 		# (seq_len, 9)
# 	sim_x = sim_x[:-1]
# 	sur_x = sur_x[:-1]
# 	sim_y = P2Cauchy(sim_y[1:], F)
# 	sur_y = P2Cauchy(sur_y[1:], F)

# 	# timestep refinement
# 	refine_y_pred = kwargs.get('refine_y_pred', None)
# 	refine_x = kwargs.get('refine_x', None)
# 	refine_F = kwargs.get('refine_F', None)
# 	refine_y_pred = P2Cauchy(refine_y_pred, refine_F)

# 	if shift is not None:
# 		sim_y[0] += shift.flatten(order)
# 		sur_y[0] += shift.flatten(order)

# 	fig, axes = plt.subplots(*names.shape, figsize=figsize, tight_layout=True)
# 	if allInOne:
# 		for i, label in enumerate(names.flatten()):
# 			lines = axes.plot(sim_x, sim_y[:,i], '--', alpha=0.8)
# 			axes.plot(sur_x, sur_y[:,i], c=lines[0].get_color(), label=f'{label}', alpha=0.8)
# 	else:
# 		if isinstance(axes, plt.Axes):
# 			axes = np.array([axes])
# 		for i, (ax, name) in enumerate(zip(axes.flatten(order), names.flatten(order))):
# 			i = 1
# 			ax.plot(sim_x, sim_y[:,i], label='simulation', c='cyan', alpha=0.8)
# 			ax.plot(sur_x, sur_y[:,i], label='surrogate', c='salmon', alpha=0.8)
# 			ax.plot(refine_x, refine_y_pred[:,i], label='surrogate_refine', c='green', alpha=0.8)
# 			ax.set_xlabel('loadstep')
# 			ax.set_ylabel(name)
# 	plt.legend(loc=(1.1, 0))
# 	return fig