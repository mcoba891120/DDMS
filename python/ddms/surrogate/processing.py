import numpy as np
import torch
import damask
import yaml
import glob
import warnings
import os
import stat
from threading import Timer
from tqdm import tqdm
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from typing import Dict, Union, Optional
from . import tensor as _tensor
from . import mechanics as _mechanics

class RepeatedTimer():
	def __init__(self, interval, function, **kwargs):
		self._timer     = None
		self.interval   = interval
		self.function   = function
		self.kwargs     = kwargs
		self.is_running = False

	def _run(self):
		self.is_running = False
		self.start()
		self.function(**self.kwargs)

	def start(self):
		if not self.is_running:
			self._timer = Timer(self.interval, self._run)
			self._timer.start()
			self.is_running = True

	def stop(self):
		self._timer.cancel()
		self.is_running = False
		print('timer stopped !')

class RemoteWatcher():
	"""
	ssh & sftp to download/remove data on host, 
	useful while generating training data on Zinfandel
	"""
	def __init__(self, hostname, username, password, port) -> None:
		import paramiko
		try:
			self.sshClient = paramiko.SSHClient()
			self.sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())
			self.sshClient.connect(hostname, port, username, password)
			self.sftpClient = paramiko.SFTPClient.from_transport(self.sshClient.get_transport())
			print('connected !')
		except Exception:
			raise ConnectionError('ssh connection failed !')
		
		self.busy = False
		self.queue = []

	def exec_command(self, command):
		return self.sshClient.exec_command(command) 
	
	def add_queue(self, remote_path, local_path):
		if not (remote_path, local_path) in self.queue:
			self.queue.append((remote_path, local_path))

	def exec_queue(self):
		if not self.busy:
			self.busy = True
			while self.queue:
				self.download(*self.queue.pop(0))
		self.busy = False
		print('watcher idling...')

	def download(self, remote_path, local_path, remove=True):
		if not self.exists_remote(remote_path):
			return
		
		remote_name = remote_path.split('/')[-1]
		if not stat.S_ISDIR(self.sftpClient.stat(remote_path).st_mode):
			print(f'downloading {remote_name}')
			self.sftpClient.get(remote_path, local_path)
		else:
			if not os.path.exists(local_path):
				os.mkdir(local_path)
			for name in self.sftpClient.listdir(remote_path):
				self.download(f'{remote_path}/{name}', f'{local_path}/{name}')
				
		if remove:
			print(f'removing {remote_name}')
			self.remove(remote_path)

	def remove(self, remote_path, force=False):
		if self.exists_remote(remote_path):
			if stat.S_ISDIR(self.sftpClient.stat(remote_path).st_mode):
				if force:
					self.exec_command(f'rm -rf {remote_path}')
				else:
					self.sftpClient.rmdir(remote_path)
			else:
				self.sftpClient.remove(remote_path)

	def exists_remote(self, remote_path):
		try:
			self.sftpClient.stat(remote_path)
		except FileNotFoundError:
			return False
		return True
	
	def finish(self):
		self.sshClient.close()
		self.sftpClient.close() 
		print('conncection closed !')

class Args():
	def __init__(self, **kwargs) -> None:
		# default value
		self.mode 			= 'test'
		self.exp_name       = ''
		self.run_name       = ''
		self.run_id         = ''
		self.save_root      = ''
		self.data_path      = ''
		self.specify_data 	= ['']
		self.epochs			= 10000
		self.batch_size		= 50
		self.learning_rate	= 0.0001
		self.device			= 'cuda'
		self.sync 			= 'online'
		self.backend		= 'wandb'

		# overwrite if given
		for k, v in kwargs.items():
			setattr(self, k, v)

class Scaler():
	def __init__(self, along={}, scale_type='standard', data_scale={}):
		assert scale_type in ['minmax', 'standard', 'lmsc'], f'scale type should be one of "minmax" or "standard" !'
		self.along = along
		self.scale_type = scale_type
		self.data_scale = data_scale

		# if along=={}:
		# 	warnings.warn('got empty `along`, some scaling behaviors may be incorrect!')

	def fit(self, data:Data) -> None:
		for k in self.along.keys():
			self.data_scale[k] = torch.tensor([[float('inf'), float('-inf')]]*data[k].size(-1))		# (feature_dim, 2)
		
		for k in self.along.keys():
			for s in self.along[k]:
				if self.scale_type=='minmax':
					self.data_scale[k][s,0] = data[k][...,s].min()									# (slice_feature_dim, )
					self.data_scale[k][s,1] = data[k][...,s].max()									# (slice_feature_dim, )
				if self.scale_type=='standard':
					self.data_scale[k][s,0] = data[k][...,s].mean()									# (slice_feature_dim, )
					self.data_scale[k][s,1] = data[k][...,s].std()									# (slice_feature_dim, )
				if self.scale_type=='lmsc':
					self.data_scale[k][s,0] = torch.zeros(1)										# (slice_feature_dim, )
					self.data_scale[k][s,1] = data[k][...,s].std()									# (slice_feature_dim, )

		# sanity check
		for v in self.data_scale.values():
			assert not torch.any(torch.isinf(v)), f'data_scale contains `inf` value !'

	def transform(self, data:Union[Data,Tensor], key:Optional[str]=None,
		   				strict:Optional[bool]=True) -> Union[Data,Tensor]:
		if type(data)==Data:
			for k in self.along.keys():
				data[k] = data[k].sub(self.loc[k]).div(self.scale[k])
			return data
		
		if type(data)==Tensor and key:
			if not strict:
				# use the first element of `data_scale`
				return data.sub(self.loc[key][0]).div(self.scale[key][0])
			return data.sub(self.loc[key]).div(self.scale[key])
	
	def inverse_transform(self, data:Union[Data,Tensor], key:Optional[str]=None,
			   					strict:Optional[bool]=True) -> Union[Data,Tensor]:
		if type(data)==Data:
			for k in self.along.keys():
				data[k] = data[k].mul(self.scale[k]).add(self.loc[k])
			return data
		
		if type(data)==Tensor and key:
			if not strict:
				# use the first element of `data_scale`
				return data.mul(self.scale[key][0]).add(self.loc[key][0])
			return data.mul(self.scale[key]).add(self.loc[key])

	def fit_transform(self, data:Data) -> Data:
		self.fit(data)
		return self.transform(data)

	def to(self, device):
		for k in self.data_scale.keys():
			self.data_scale[k] = self.data_scale[k].to(device)
		return self

	@property
	def loc(self):
		return {k: v[...,0] for k, v in self.data_scale.items()}
	
	@property
	def scale(self):
		if self.scale_type=='minmax':
			return {k: v[...,1]-v[...,0] for k, v in self.data_scale.items()}
		if self.scale_type in ['standard', 'lmsc']:
			return {k: v[...,1] for k, v in self.data_scale.items()}
		
	def state_dict(self):
		return {
			'along': self.along,
			'scale_type': self.scale_type,
			'data_scale': self.data_scale
		}

	def load_state_dict(self, state_dict, map_location='cpu'):
		for k, v in state_dict.items():
			setattr(self, k, v)
		return self.to(map_location)

class Result():
	""" need optimized for `targets`, use targets dictionary ? """
	def __init__(self, hdf5_path, targets=['F_e', 'F_p', 'P', 'dPdF'], remove_first=True):
		self.targets = targets
		self.data = damask.Result(hdf5_path).get(targets)
		if remove_first:
			self.data.pop('increment_0')

	def get(self, target:str) -> Tensor:
		if target not in self.targets:
			raise ValueError(f'target {target} is not in targets of {self.targets}!')

		val = []
		for inc in self.data.values():
			# single self.targets
			if isinstance(inc, np.ndarray):
				val.append(inc)
			
			# multiple self.targets
			if isinstance(inc, dict):
				if target in inc.keys():
					val.append(inc[target])
				else:					
					if target=='dPdF':	
						val.append(inc['homogenization'])
					else:
						val.append(inc['phase'][target])

		return torch.from_numpy(np.array(val))		# (seq_len, num_grid, *)

class ScriptData(Module):
	def __init__(self, data_scale:Dict[str,Tensor], kwargs:Optional[Dict[str,Tensor]]={}):
		super().__init__()
		for k, v in data_scale.items():
			setattr(self, f'{k}_sc', v)
		for k, v in kwargs.items():
			setattr(self, k, v)

def norm(x, mm):
	return x.sub(mm[...,0]).div(mm[...,1]-mm[...,0])

def denorm(x, mm):
	return x.mul(mm[...,1]-mm[...,0]).add(mm[...,0])


def get_loadstep(yaml_path):
	""" 
	get loadstep from `load.yaml`
	
	### WARNING: 
	- use `get_loadstep_fromhdf5` instead
	"""
	with open(yaml_path, 'r') as stream:
		load = yaml.safe_load(stream)
		loadsteps = load['loadstep']                                    	# (loadstep_len, {})

	Fs = []
	for i, loadstep in enumerate(loadsteps):
		BC 			= loadstep['boundary_conditions']
		discret 	= loadstep['discretization']
		fout 		= loadstep['f_out']
		mechanical 	= BC['mechanical']

		inc_per_out = discret['N']//fout
		sec_per_out = discret['t']/discret['N']*fout

		if 'F' in mechanical:
			assert inc_per_out==1
			Fs.append(mechanical['F'])
		elif 'dot_F' in mechanical:
			dot_F = [[c if c!='x' else 0 for c in r] for r in mechanical['dot_F']]							# (3, 3), TODO WARNING: if 'x' BC_F==0? 
			dot_F = [[[c*sec_per_out*seq for c in r] for r in dot_F] for seq in range(1, inc_per_out+1)] 	# (inc_per_out, 3, 3)
			Fs.extend(dot_F)
		else:
			raise ValueError('not supported boundary condition !')

	Fs = torch.tensor(Fs)
	Fs = Fs.flatten(-2,-1).unsqueeze(0)														# (1, seq_len, 9)
	return Fs

def get_loadstep_fromhdf5(hdf5_path):
	"""
	get homogenized deformation gradient from DAMASK hdf5, using `F_e` and `F_p`
	"""
	res = damask.Result(hdf5_path)

	# F = FeFp
	Fe = np.array([fe for fe in res.get('F_e').values()])	# (seq_len+1, num_grid, 3, 3)
	Fp = np.array([fp for fp in res.get('F_p').values()])	# (seq_len+1, num_grid, 3, 3)
	F = np.einsum('snij, snjk -> snik', Fe, Fp)				# (seq_len+1, num_grid, 3, 3)
	F = torch.from_numpy(F.mean(axis=1))[1:]				# (seq_len, 3, 3), remove increment_0
	return F.reshape(1, -1, 9)								# (1, seq_len, 9)


def get_UCS_data(folder_path, result_path, check_len=101) -> Data:
	"""
	get U-CS data, see https://doi.org/10.1016/j.jmps.2021.104668
	- folder_path: data.pt that contains `euler` angles
	- result_path: *.hdf5 that contains strain and stress
	"""
	folder = folder_path.split('/')[-1]
	hdf5_path = glob.glob(f'{result_path}/*.hdf5')[0]
	
	data 	= torch.load(f'{folder_path}/data.pt')
	euler 	= data.euler[1:].permute(1,0,2) 								# (num_node, seq_len, 3)
	if euler.size(1) != check_len:
		print(f'{folder} get seq_len {euler.size(0)} !')
		return None

	res 	= Result(hdf5_path, ['F_e', 'F_p', 'P', 'dPdF'])
	Fe 		= res.get('F_e')
	Fp 		= res.get('F_p')
	F		= _mechanics.FeFp_to_F(Fe, Fp)												# (seq_len, num_grid, 3, 3)
	P 		= res.get('P')													# (seq_len, num_grid, 3, 3)
	dPdF 	= res.get('dPdF')												# (seq_len, num_grid, 81)
	R, U 	= _tensor.polar(F)													# (seq_len, num_grid, 3, 3)

	# TODO CS needs to be rotate !!
	CS 		= _mechanics.P_to_CS(P, F)													# (seq_len, num_grid, 6)
	dCSdE 	= _mechanics.dPdF_to_dCSdE(dPdF, P, F)										# (seq_len, num_grid, 6, 6)

	# average
	U = _tensor.sym33_to_m6(U, MAP=_tensor.MAP.m33toEXP6).mean(dim=1).unsqueeze(0)		# (1, seq_len, 6)
	CS = CS.mean(dim=1).unsqueeze(0)										# (1, seq_len, 6)
	dCSdE = dCSdE.mean(dim=1).flatten(-2, -1).unsqueeze(0)					# (1, seq_len, 36)

	return Data(
		euler=euler.float(),												# (num_node, seq_len, 3)
		U=U.float(), 														# (1, seq_len, 6)
		CS=CS.float(),														# (1, seq_len, 6)
		dCSdE=dCSdE.float(),												# (1, seq_len, 36)
		num_node=euler.size(0),
		desc=folder,
	)

def get_dev5_data(result_path, check_len=101) -> Data:
	"""
	get dHSdev5-CSdev5 data, see https://doi.org/10.1016/j.ijplas.2022.103430
	- result_path: *.hdf5 that contains strain and stress
	"""
	folder = result_path.split('/')[-1]
	try:
		hdf5_path = glob.glob(f'{result_path}/*.hdf5')[0]
		res = Result(hdf5_path, ['F_e', 'F_p', 'P', 'dPdF'])
	except (IndexError, OSError):
		print(f'{folder} get result failed !')
		return None

	if check_len and len(res.data) != check_len:
		print(f'{folder} get seq_len {len(res.data)} !')
		return None

	Fe 		= res.get('F_e')
	Fp 		= res.get('F_p')
	P 		= res.get('P')													# (seq_len, num_grid, 3, 3)
	dPdF 	= res.get('dPdF')												# (seq_len, num_grid, 81)
	
	F		= _mechanics.FeFp_to_F(Fe, Fp)												# (seq_len, num_grid, 3, 3)
	CS 		= _mechanics.P_to_CS(P, F)													# (seq_len, num_grid, 3, 3)
	dCSdE 	= _mechanics.dPdF_to_dCSdE(dPdF, P, F, MAP=_tensor.MAP.m3333toIMP66)				# (seq_len, num_grid, 6, 6)
	result 	= _mechanics.FCS_to_dev5(F, CS, dim=0)

	# average
	dHS 	= result['dHS'].mean(dim=1).unsqueeze(0)						# (1, seq_len-1, 6)
	dHSdev5 = result['dHSdev5'].mean(dim=1).unsqueeze(0)					# (1, seq_len-1, 5)
	dHShyd6 = result['dHShyd6'].mean(dim=1).unsqueeze(0)					# (1, seq_len-1, 6)
	CS 		= result['CS'].mean(dim=1).unsqueeze(0)							# (1, seq_len-1, 6)
	CSdev5 	= result['CSdev5'].mean(dim=1).unsqueeze(0)						# (1, seq_len-1, 5)
	CShyd6 	= result['CShyd6'].mean(dim=1).unsqueeze(0)						# (1, seq_len-1, 6)
	dCSdE 	= dCSdE.mean(dim=1).flatten(-2, -1).unsqueeze(0)				# (1, seq_len, 36)

	return Data(
		dHS=dHS.float(),
		dHSdev5=dHSdev5.float(),
		dHShyd6=dHShyd6.float(),
		CS=CS.float(),
		CSdev5=CSdev5.float(),
		CShyd6=CShyd6.float(),
		dCSdE=dCSdE.float(),
		num_node=1,
		desc=folder
	)

# >--------------------------------------------------
# > data augmentation 
# >--------------------------------------------------
def flat_augmenter(src_data_list, aug_len=10, fnt_p=0.5) -> Data:
	""" 
	augment "flat" values at time `aug_t` with length of `aug_len`,
	if `aug_t` is `None`, will use random time
	"""
	def augment(src_data, aug_t):
		aug_data = Data()
		for k, v in src_data.to_dict().items():
			if k=='num_node' or k=='desc': 
				aug_data[k] = src_data[k]
				continue
			src = src_data[k].permute(1,0,2)								# (s, b, c)
			tgt = src[aug_t].repeat(aug_len,1,1)							# (aug_len, b, c)
			aug = torch.concat([src[:aug_t], tgt, src[aug_t:]], dim=0)		# (s+aug_len, b, c)
			aug_data[k] = aug.permute(1,0,2)								# (b, s+aug_len, c)
		return aug_data

	seq_len = src_data_list[0].U.size(1)
	for src_data in src_data_list:
		aug_ts = torch.randint(1, seq_len, size=(2,))				# augment time

		# flat augment
		aug_t = aug_ts[0]
		aug_data = augment(src_data, aug_t)

		# front augment
		aug_t = 0 if torch.rand(1) <= fnt_p else aug_ts[1]
		aug_data = augment(aug_data, aug_t)
		yield aug_data

def noise_augmenter(src_data, lmb=None, dny_p=0) -> Data:
	"""
	add noise to original data input, i.e., `deps`
	with probability of `dny_p` to deny the noise for each component
	"""
	lmb = lmb or src_data['deps'].mean() * 1e-3								# noise magnitude
	print(f'running noise_augmenter with magnitude of {lmb:e}')

	noise = torch.rand(src_data['deps'].size())*2-1							# [-1, 1)
	noise = noise.mul(lmb)													# [-lmb, lmb)

	for b, n in enumerate(noise):
		deny_mask = torch.rand(noise.size(-1)) <= dny_p
		noise[b,...,deny_mask] = 0

	aug_data = src_data.clone()
	aug_data['deps'] = aug_data['deps'].add(noise)
	return aug_data


# >--------------------------------------------------
# > resume existing data & transformation
# > WARNING: not optimized, carefully use !
# >--------------------------------------------------
def reconstruct_data_list(src_data, src_slice) -> list:
	""" reconstruct data_list from torch_geometric Data """
	num_data = len(src_slice['desc']) - 1
	data_list = []
	for i in tqdm(range(num_data), desc='reconstruct data_list'):
		data = Data()
		for k, v in src_slice.items():
			if k=='desc' or k=='num_node':
				slc = i
			else:
				slc = slice(v[i], v[i+1])
			data[k] = src_data[k][slc]
		data_list.append(data)
	return data_list

def get_sep_from_all(data_root):
	"""
	get `seperate` minmax from `all` minmax 
	"""
	data_normed = torch.load(f'{data_root}/LSTM_U_CS.pt')[0]
	data_minmax_all = torch.load(f'{data_root}/LSTM_U_CS_minmax.pt')

	data_minmax_sep = data_minmax_all.copy()
	for k in data_minmax_all.keys():
		Min = denorm(data_normed[k], data_minmax_all[k]).amin(dim=(0,1))
		Max = denorm(data_normed[k], data_minmax_all[k]).amax(dim=(0,1))
		data_minmax_sep[k] = torch.stack([Min, Max], dim=1)

	torch.save(data_minmax_sep, f'{data_root}/LSTM_U_CS_minmax_sep.pt')

def get_dev_from_UCS(src_data, src_slices, data_root, dst_name, noise=False, data_standard=None):
	"""
	get `dev`(deviatoric) data from `U_CS` data
	ref dev: https://doi.org/10.1016/j.ijplas.2022.103430
	ref U_CS: https://doi.org/10.1016/j.jmps.2021.104668
	"""
	U 		= src_data.U.double()									# EXP
	CS 		= src_data.CS.double()									# IMP
	result 	= _mechanics.UCS_to_dev5(U, CS)
	dHSdev5 = result['dHSdev5']
	CSdev5 	= result['CSdev5']
	CShyd6 	= result['CShyd6']

	# sanity check 
	# CS_ = reconstruct_CS6_from_CSdev5(CSdev5)
	# print(sigma_dev[0,0,MAP_33toIMP6[0], MAP_33toIMP6[1]])
	# print(CS_[0,0])
	# print((sigma_dev[:,:,MAP_33toIMP6[0], MAP_33toIMP6[1]] - CS_).max())

	data = Data(
		euler=src_data.euler,
		deps=dHSdev5.float(),
		CSdev5=CSdev5.float(),
		CShyd=CShyd6.float(),
		dCSdE=src_data.dCSdE[:,1:,:],
		num_node=src_data.num_node,
		desc=src_data.desc
	)
	slices = {
		'euler': src_slices['euler'],
		'deps': src_slices['U'],
		'CSdev5': src_slices['CS'],
		'CShyd': src_slices['CS'],
		'dCSdE': src_slices['dCSdE'],
		'num_node': src_slices['num_node'],
		'desc': src_slices['desc']
	}

	get_standard = lambda x: torch.tensor([x.mean(), x.std()]).repeat(x.size(-1), 1)
	data_standard = data_standard or {
		'euler': get_standard(data.euler),	# WARNING: weird
		'deps': get_standard(data.deps),
		'CSdev5': get_standard(data.CSdev5),
		'CShyd': get_standard(data.CShyd),
		'dCSdE': get_standard(data.dCSdE),
	}
	for k in data_standard.keys():
		data[k] = norm(data[k], data_standard[k])

	# data augmentation
	if noise:
		data = noise_augmenter(data, lmb=3e-4, dny_p=0.5)

	torch.save((data, slices), f'{data_root}/{dst_name}.pt')
	torch.save(data_standard, f'{data_root}/{dst_name}_standard.pt')


def get_amplitude(X0:Tensor, result_path:str) -> Tensor:
	"""
	get abaqus amplitude from loading path of `result_path`, 
	used in comparison of `damask simulation` & `abaqusNN single element test`
	- X0, xyz coordinate 
	- X0.shape == (*, 3)
	"""
	res = Result(result_path, ['F_e', 'F_p'])
	F = _mechanics.FeFp_to_F(res.get('F_e'), res.get('F_p')).mean(dim=1)	# (s, 3, 3)
	return torch.einsum('sij, ...j -> ...si', F, X0)
