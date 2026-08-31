import damask
import numpy as np
import pandas as pd
import matplotlib
try:
	matplotlib.use('tkAgg')
except ImportError:
	matplotlib.use('Agg')	# no GUI backend available (e.g. headless/CI)
import matplotlib.pyplot as plt
import os
from scipy.linalg import polar
from scipy.stats import qmc
from scipy.interpolate import CubicSpline
from orix.quaternion import Orientation, symmetry
from orix.sampling import get_sample_fundamental
from sklearn.cluster import DBSCAN
from intersect import intersection

# >----------------------------------------------------------------------------------------------------
# > Loading
# >----------------------------------------------------------------------------------------------------
def get_loadstep(mechanical:dict, t, N, f_out):
	return {
		'boundary_conditions': {
			'mechanical': mechanical
		},
		'discretization': {
			't':t,
			'N':N
		},
		'f_out': f_out 	
	}

def get_loadsteps_LHS(n=3, F_min=-0.01, F_max=0.01, t_max=70+1, save_root=None, seed=0):
	"""
	Info: Latin hypercube sampling (LHS)
		2 dimensions (time & Fij)
		3+2 points (3 LHS + first + last)
	Info: CubicSpline
		create smooth spline with LHS samples

	Warning:
		legacy code.

	Args:
		n: The number of LHS samples.
		F_min: The minimum value of Fij.
		F_max: The maximum value of Fij.
		t_max: The total loading time.
		save_root: The root directory to save the plot.
		seed: The random seed of LHS sampler.

	Returns:
		A list of load steps.
	"""
	def _scale(F_fit, F_sample, min_max):
		"""
		scale spline & LHS with min_max
		"""
		Min, Max = F_fit.min(), F_fit.max()
		scale = lambda x: x/Max*min_max[1] if abs(Max)>abs(Min) else x/Min*min_max[0]
		F_fit = scale(F_fit)
		F_sample = scale(F_sample)
		return F_fit, F_sample

	sampler = qmc.LatinHypercube(d=2, seed=seed) 								# d:(t, F), LHS sampling

	t_fits = np.arange(0, t_max); F_fits = []									# t includes first and last
	t_samples = []; F_samples = []
	for _ in range(9):															# 9 components of deformation gradient
		xys = sampler.random(n=n) 												# (n, d)
		xys = qmc.scale(xys, [t_max//4-1, F_min], [t_max//4*3-1, F_max])		# t excludes first and last
		xys = xys[np.argsort(xys[:,0])]
		xys = np.concatenate([[[0,0]], xys, [[t_max-1,0]]], axis=0) 			# (n+2, 2), add first and last

		spl = CubicSpline(xys[:, 0], xys[:, 1], bc_type='periodic')
		F_fit, F_sample = _scale(spl(t_fits), xys[:,1], [F_min, F_max])

		t_samples.append(xys[:, 0])
		F_samples.append(F_sample)
		F_fits.append(F_fit)
	
	F_fits = np.reshape(F_fits, (9, -1)).T.reshape(-1,3,3)+np.eye(3)			# (t, 3, 3)
	t_samples = np.reshape(t_samples, (9, -1)).T.reshape(-1,3,3)				# (n+2, 3, 3)
	F_samples = np.reshape(F_samples, (9, -1)).T.reshape(-1,3,3)+np.eye(3)		# (n+2, 3, 3)

	# get load step
	loadsteps = []
	for i, F_fit in enumerate(F_fits[1:]):
		mechanical = {'F': F_fit.tolist()}
		t = int(10*np.linalg.norm(F_fit)) 										# to keep constant rate
		N = 15 if i<4 else 10
		f_out = 15 if i<4 else 10
		loadstep = get_loadstep(mechanical, t, N, f_out)
		loadsteps.append(loadstep)
	
	# plot load
	if save_root:
		fig, ax = plt.subplots(tight_layout=True)
		for i in range(3):
			for j in range(3):
				shift = -1 if i==j else 0
				ax.plot(t_fits, F_fits[:,i,j]+shift, label=f'E{i+1}{j+1}')
		ax.legend()
		plt.savefig(f'{save_root}/loadsteps_LHS.png')
	return loadsteps

def get_loadsteps_RW(N=10, t_max=100, U_max=0.01/3, save_root=None, seed=0):
	"""
	Create complex loading steps by random walk(RW) & Hermite polynomial. 
	Modified from https://doi.org/10.1016/j.jmps.2021.104668.

	Args:
		N: The number of RW steps.
		t_max: The total loading time.
		U_max: The max strain increment factor -> strain rate ~ 0.001.
		save_root: The root directory to save the plot.
		seed: The random seed.

	Returns:
		A list of load steps.
	"""
	def _get_path(X, N, t_max, U_max, deg=5, fit_pts=[0,2,7,10]):
		t = np.linspace(0, t_max, N+1)
		U_n = [0]
		for _ in range(N):
			dt_n = t_max//N 														# assuming uniform
			v_n = np.random.choice([-1, 0, 1], size=1, p=[1/3, 1/3, 1/3])[0]
			U_n += [U_n[-1] + v_n*U_max*np.sqrt(dt_n)]
		U_n = np.array(U_n)

		hermite = np.polynomial.Hermite.fit(t[fit_pts], U_n[fit_pts], deg)
		return hermite(X), U_n

	np.random.seed(seed=seed)

	step = 1
	X = np.arange(0, 101, step)
	X_n = np.linspace(0, 101, N+1)
	U = np.zeros((3, 3, len(X)))
	U_n = np.zeros((3, 3, N+1))

	# TODO: verification required, U (stretch tensor, symmetric F)
	U[0, 0], U_n[0, 0] = _get_path(X, N, t_max, U_max)
	U[1, 1], U_n[1, 1] = _get_path(X, N, t_max, U_max)
	U[2, 2], U_n[2, 2] = _get_path(X, N, t_max, U_max)
	U[1, 2], U_n[1, 2] = U[2, 1], U_n[2, 1] = _get_path(X, N, t_max, U_max)
	U[0, 2], U_n[0, 2] = U[2, 0], U_n[2, 0] = _get_path(X, N, t_max, U_max)
	U[0, 1], U_n[0, 1] = U[1, 0], U_n[1, 0] = _get_path(X, N, t_max, U_max)

	U = np.einsum('ijk->kij', U) + np.eye(3) 										# (seq_len, 3, 3)
	U_n = np.einsum('ijk->kij', U_n) + np.eye(3)  									# (seq_len, 3, 3)

	# construct loadsteps
	loadsteps = []
	for i, u in enumerate(U):
		dt_n = t_max//N																# assuming uniform, TODO: WARNING should be t_max//len(X), i.e., dt_n=1
		N_inc = 20 if i<4 else 10													# N increment
		f_out = 2 if i<4 else 1														# 
		mechanical = {'F': u.tolist()}												# (3, 3)
		loadsteps.append(get_loadstep(mechanical, dt_n, N_inc, f_out))

	if save_root:
		# plot load in Green-Lagrange strain
		E = 0.5*(np.einsum('sji,sjk->sik', U, U) - np.eye(3))
		E_n = 0.5*(np.einsum('sji,sjk->sik', U_n, U_n) - np.eye(3))
		
		fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
		ax.plot(X, E.flatten(-2, -1), label=['E11','E22','E33','E23','E13','E12'])
		ax.scatter(X, E_n.flatten(-2, -1))
		ax.legend(loc=(1.02, 0))
		ax.set_xlabel('loadstep')
		plt.savefig(f'{save_root}/loadsteps_RW.png')
	return loadsteps

def get_loadsteps_rotshear(save_root=None):
	""" 
	Create shearxy loading steps with different rotation angles.
	Modified from https://doi.org/10.1016/j.ijplas.2022.103430 

	Warning:
		legacy code.
	"""
	N = 10												# N interval
	t_max = 100											# total time, 100 sec
	gamma_max = 0.1										# maximum value of shear

	step = 1
	X = np.arange(0, 101, step)
	F = np.array([np.eye(3) for _ in X])
	F[:,0,1] = np.linspace(0, gamma_max, len(X))
	U = [polar(f)[1] for f in F]

	theta = 45
	R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
				  [np.sin(np.radians(theta)),  np.cos(np.radians(theta)), 0],
				  [		 				   0, 						   0, 1]])
	F = np.einsum('ji, sjk, kl -> sil', R, F, R)

	# construct loadsteps
	loadsteps = []
	for i, u in enumerate(U):
		dt_n = t_max//N									# assuming uniform, TODO: WARNING should be t_max//len(X), i.e., dt_n=1
		N_inc = 15 if i<4 else 10						# N increment
		f_out = 15 if i<4 else 10						# 
		mechanical = {'F': u.tolist()}					# (3, 3)
		loadsteps.append(get_loadstep(mechanical, dt_n, N_inc, f_out))

	if save_root:
		# plot load in Green-Lagrange strain
		E = 0.5*(np.einsum('sji,sjk->sik', U, U) - np.eye(3))
		
		fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
		ax.plot(X, E.flatten(-2, -1), label=['E11','E22','E33','E23','E13','E12'])
		ax.legend(loc=(1.02, 0))
		ax.set_xlabel('loadstep')
		plt.savefig(f'{save_root}/load_rotshear.png')
	return loadsteps

# >----------------------------------------------------------------------------------------------------
# > Texture
# >----------------------------------------------------------------------------------------------------
def get_weight(texture, num_grain):
	"""
	Get Euler angles, weights and sigmas to establish ODF in dream3d.

	Args:
		texture: The type of texture, can be `singleCrystal`, `biCrystal`, `random`, or `rolled`.
		num_grain: The number of grains.

	Returns:
		e1, e2, e3: The random orientations.
		sigma: The random grain sizes.
		weight: The weights of the random orientations.
	"""
	e1 = np.random.rand(num_grain)*np.pi*2
	e2 = np.random.rand(num_grain)*np.pi*2
	e3 = np.random.rand(num_grain)*np.pi*2
	sigma = np.ones(num_grain)
	
	if texture=='singleCrystal':
		weight = np.zeros(num_grain)
		idx = np.random.randint(len(weight))
		weight[idx] = 500000
	if texture=='biCrystal':
		weight = np.zeros(num_grain)
		idx = np.random.randint(len(weight), size=2)
		weight[idx] = 500000
	if texture=='random':
		weight = np.ones(num_grain)
	if texture=='rolled':
		# extend + shift
		e1 = np.random.rand(num_grain)*np.pi/6 + np.random.rand(1)*(np.pi*2-np.pi/6)
		e2 = np.random.rand(num_grain)*np.pi/6 + np.random.rand(1)*(np.pi*2-np.pi/6)
		e3 = np.random.rand(num_grain)*np.pi/6 + np.random.rand(1)*(np.pi*2-np.pi/6)
		sigma = np.ones(num_grain)*5
		weight = np.ones(num_grain)
	
	if 'morient' in texture:
		euler = get_sample_fundamental(10, symmetry.Oh).to_euler()
		e1, e2, e3 = euler[:,0],euler[:,1],euler[:,2]
		sigma = np.ones(len(e1))
		weight = np.ones(len(e1))
	
	return e1, e2, e3, sigma, weight

def get_MOEP(quats: np.array) -> np.array:
	"""
	Get modified orientation evolution path(MOEP).
	See https://doi.org/10.1016/j.cma.2021.114392.

	Args:
		quats: The quaternions of the evolution path. (inc, num_node, 4)

	Returns:
		MOEP: The modified orientation evolution path.
	"""
	class MOrientation(Orientation):
		"""modified Orientation class"""
		def __init__(self, data, symmetry=None):
			super().__init__(data, symmetry)
		
		def equivalent(self, grain_exchange=False):
			"""for FCC symmetry"""
			equivalent = super().equivalent(grain_exchange)
			mask = np.array([1 if (i//24)%2==0 else 0 for i in range(equivalent.size)]).astype(bool)
			return equivalent[mask]

	def num_in_FZ(eulers):
		"""
		phi1: 	[0,2*pi]
		Phi: 	[0,pi/2] & 
				cos(Phi) > cos(phi2)/sqrt(1+cos(phi2)**2) &
				cos(Phi) > cos(pi/2-phi2)/sqrt(1+cos(pi/2-phi2)**2)
		phi2: 	[0,pi/2]
		"""
		phi1, Phi, phi2 = eulers[:,0], eulers[:,1], eulers[:,2]
		check_matrix = np.zeros_like(eulers).astype(bool)
		check_matrix[:,0] = np.logical_and(phi1>=0, phi1<=2*np.pi)
		check_matrix[:,1] = np.logical_and(Phi>=0, Phi<=np.pi/2)
		check_matrix[:,1] = np.logical_and(check_matrix[:,1], np.cos(Phi)>np.cos(phi2)/np.sqrt(1+np.cos(phi2)**2))
		check_matrix[:,1] = np.logical_and(check_matrix[:,1], np.cos(np.pi/2-phi2)/np.sqrt(1+np.cos(np.pi/2-phi2)**2))
		check_matrix[:,2] = np.logical_and(phi2>=0, phi2<=np.pi/2)
		return np.count_nonzero(np.all(check_matrix, axis=1))

	inc, num_node, _ = quats.shape
	MOEP = np.zeros((inc, num_node, 3))

	for n in range(num_node):
		OEP = MOrientation(quats[:,n,:], symmetry.Oh).equivalent().to_euler() 			# (24*inc, 3)
		labels = DBSCAN(eps=1, min_samples=1).fit_predict(OEP) 							# (24*inc)
		ulabels, counts = np.unique(labels, return_counts=True)
		ulabels = ulabels[counts==inc] 													# unique labels of full continuous OEP

		max_num = 0
		for ulabel in ulabels:
			eulers = OEP[labels==ulabel] 												# (inc, 3)
			num = num_in_FZ(eulers)
			if num > max_num:
				MOEP[:,n,:] = eulers
				max_num = num
	return MOEP

def num_grain_is_consistent(csv_path, check_grain=71):
	"""
	Check if the number of grains in the dream3d CSV file is consistent with the given grain number.

	Args:
		csv_path: The path to the CSV file.
		check_grain: The expected number of grains.

	Returns:
		True if the number of grains is consistent, False otherwise.
	"""
	if not os.path.exists(csv_path):
		return False
	
	with open(csv_path, 'r') as f:
		num_grain = int(f.readline())
	return True if num_grain==check_grain else False

# >----------------------------------------------------------------------------------------------------
# > Graph
# >----------------------------------------------------------------------------------------------------
def get_node_features(grid, result, csv_path):
	"""
	Get node features, including texture(Euler angles), number of neighbor, and volume of a grain.

	Args:
		grid: damask geometry file(.vti)
		result: damask result file(.hdf5)
		csv_path: dream3d output file(.csv)

	Info:
		node features are all sorted by `FeatureIDs` from Dream3d.

	Returns:
		eulers: The Euler angles of the texture.
		num_neighbor: The number of neighbors of a grain.
		volume: The volume of a grain.
	"""
	quats = []
	material = grid.material 
	grainIDs = np.unique(material)
	num_node = len(grainIDs)
	for Fe_t in result.get('F_e').values():
		for grainID in grainIDs:
			mask 	= material.flatten(order='F')==grainID
			Fe 		= Fe_t[mask]
			R = [polar(fe)[0].T for fe in Fe]
			o = damask.Orientation.from_matrix(R=R, family='cubic', lattice='cF').average()
			quats.append(o.as_quaternion())
	
	df = pd.read_csv(csv_path, header=1, index_col=False, nrows=num_node)
	num_neighbor = df['NumNeighbors'].values.reshape(-1, 1) 					 	# (num_node, 1)
	volume = df['Volumes'].values.reshape(-1,1) 									# (num_node, 1)

	quats = np.reshape(quats, (-1, num_node, 4)) 									# (inc, num_node, 4)
	eulers = get_MOEP(quats)
	return eulers, num_neighbor, volume

def get_edge_features(grid, csv_path):
	"""
	Get edge features, including edge index, suface areas and surface normal vector.

	Info:
		node features are all sorted by `FeatureIDs` from Dream3d.

	Returns:
		edge_index: The edge indices.
		surface_areas: The surface areas of the edges.
		normals: The unit normal vectors between grains surface.
	"""
	num_node = grid.N_materials
	with open(csv_path, 'r') as f:
		num_neighbor_range = (num_node+3, num_node*2+3)									# dream3d output csv format
		surface_area_range = (num_node*2+4, num_node*3+4)								# dream3d output csv format

		neighbor_list = []
		area_list = []
		for i, line in enumerate(f):
			if i >=num_neighbor_range[0] and i<num_neighbor_range[1]:
				line = line.rstrip('\n').split(',')
				line = list(map(lambda x: int(x)-1, line)) 								# convert to int and change featureID to index
				neighbor_list.append(line[2:])

			if i>=surface_area_range[0] and i<surface_area_range[1]:
				line = line.rstrip('\n').split(',')
				line = list(map(float, line))
				area_list.append(line[2:])

	edge_index = []
	surface_areas = []
	normals = [] 																		# unit normal vector between grains surface
	seed_coords = damask.seeds.from_grid(grid, average=True)[0] 						# voronoi tessellation seeds, sorted by grid.material, (num_node, 3)
	for i, (neighbors, areas) in enumerate(zip(neighbor_list, area_list)):
		for n, a in zip(neighbors, areas):
			edge_index.append([i, n])
			surface_areas.append([a])
			
			n = seed_coords[n]-seed_coords[i]
			normals.append(n/np.linalg.norm(n))

	edge_index = np.array(edge_index)               									# (num_edges, 2)
	surface_areas = np.array(surface_areas) 											# (num_edges, 1)
	normals = np.array(normals) 														# (num_edges, 3)

	return edge_index, surface_areas, normals


# >----------------------------------------------------------------------------------------------------
# > Mecahinics
# >----------------------------------------------------------------------------------------------------
def get_plastic(strain, stress, YSx, YSy, axis=0, flow_stress=True):
	"""
	Get the plastic part of strain and stress.

	Args:
		strain: The engineering strain.
		stress: The engineering stress.
		YSx: The yield point x.
		YSy: The yield point y.
		axis: The axis along which the plastic strain and stress are calculated.
		flow_stress: Whether to convert the result stress to flow stress.

	Returns:
		The plastic strain and stress.
	"""
	s_idx = np.arange(stress.shape[axis])
	p_mask = np.take(stress, s_idx, axis=axis) >= YSy
	strain = np.take(strain, s_idx[p_mask], axis=axis)
	stress = np.take(stress, s_idx[p_mask], axis=axis)
	strain = np.insert(strain, 0, YSx, axis=axis)
	stress = np.insert(stress, 0, YSy, axis=axis)
	if flow_stress:
		stress -= YSy
	return strain, stress

def get_YS(strain, stress, E):
	"""
	Get the yield stress from the engineering strain and stress.

	Warning:
		Only support 1D input currently.

	Args:
		strain: The engineering strain.
		stress: The engineering stress.
		E: The Young's modulus.

	Returns:
		The yield strain and stress.
	"""
	ylim = np.max(stress)
	YSx, YSy = intersection(strain, stress, np.array([0.002, ylim/E+0.002]), np.array([0,ylim]))
	return YSx[0], YSy[0]

def get_UTS(strain, stress, axis=0):
	"""
	Get the ultimate tensile strength from the engineering strain and stress.

	Args:
		strain: The engineering strain.
		stress: The engineering stress.
		axis: The axis along which the ultimate tensile strength is calculated.

	Returns:
		The ultimate tensile strain and stress.
	"""
	uts_idx = np.arange(np.argmax(stress, axis=axis))
	strain = np.take(strain, uts_idx, axis=axis)
	stress = np.take(stress, uts_idx, axis=axis)
	return strain, stress

def get_engineering(true_strain, true_stress):
	"""
	Convert true strain and stress to engineering strain and stress.

	Args:
		true_strain: The true strain.
		true_stress: The true stress.

	Returns:
		The engineering strain and stress.
	"""
	eng_strain = np.exp(true_strain)-1
	eng_stress = true_stress/(1+eng_strain)
	return eng_strain, eng_stress

def get_logarithmic(eng_strain, eng_stress):
	"""
	Convert engineering strain and stress to true strain and stress.

	Args:
		eng_strain: The engineering strain.
		eng_stress: The engineering stress.

	Returns:
		The true strain and stress.
	"""
	true_strain = np.log(eng_strain+1)
	true_stress = eng_stress*(1+eng_strain)
	return true_strain, true_stress

def get_uniform_stress(strain, stress, l=0.0, r=0.05, num_pt=9) -> np.array:
	"""
	Get the uniformly interpolated stress values at the specified strain range.

	Args:
		strain: The strain values.
		stress: The stress values.
		l: The lower bound of the strain range.
		r: The upper bound of the strain range.
		num_pt: The number of points in the strain range.

	Returns:
		The interpolated stress values.
	"""
	pts = np.linspace(l, r, num_pt)
	return np.interp(pts, strain, stress)

def smoothing(data, window_width=1):
	"""
	Smooth the data using a moving average.

	Args:
		data: The data to be smoothed.
		window_width: The width of the moving average window.

	Returns:
		The smoothed data.
	"""
	cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
	return (cumsum_vec[window_width:]-cumsum_vec[:-window_width]) / window_width

