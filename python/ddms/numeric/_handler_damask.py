import torch
import damask
import numpy as np 
import os
import glob
import json
import subprocess 
import warnings; warnings.filterwarnings('ignore')
from torch_geometric.data import Data
from ._handler import Handler 
from .processing import (get_loadstep, get_loadsteps_RW, 
						 get_node_features, get_edge_features, get_weight, 
						 get_engineering, get_YS)
	
class HandlerDamask(Handler):
	def __init__(self, YAML, taskname, target, ROOTdir):
		super().__init__(YAML, taskname, target, ROOTdir)

	# >----------------------------------------------------------------------------------------------------
	# > preprocessing
	# >----------------------------------------------------------------------------------------------------
	def getConfigMaterial(self, dream3d='data.dream3d'):
		"""
		Get damask material.yaml from dream3d.

		Args:
			dream3d: The name of the dream3d file in `RVEdir`.
		"""
		def _correct_type(value, key):
			""" yaml somehow not accepting np.float """
			if key=='xi_0_sl':
				value = [value]
			if type(value) == np.float64:
				return float(value)
			if type(value) == list and type(value[0]) == np.float64:
				return [float(v) for v in value]
			return value

		# update material from .dream3d
		material = damask.ConfigMaterial.load_DREAM3D(f'{self.prm.RVEdir}/{dream3d}')
		material = material.material_rename_phase({'Primary': 'Aluminum'})
		material = material.material_rename_homogenization({'direct': 'SX'})
		self.template['material'] = material['material']

		# update system 
		for k in self.template['system'].keys():
			self.template['system'][k] = getattr(self.prm, k)

		# update phase
		for k in self.template['phase']['Aluminum']['mechanical']['plastic'].keys():
			self.template['phase']['Aluminum']['mechanical']['plastic'][k] = _correct_type(getattr(self.prm, k), k)

		self.template.save(f'{self.prm.WORKdir}/material.yaml')

	def getGrid(self, dream3d='data.dream3d'):
		"""
		Get damask geometry.vti from dream3d.

		Args:
			dream3d: The name of the dream3d file in `RVEdir`.
		"""
		grid = damask.Grid.load_DREAM3D(f'{self.prm.RVEdir}/{dream3d}')
		grid.save(f'{self.prm.WORKdir}/{self.prm.TASKname}')

	def getLoad(self, loadsteps:list=None):
		"""
		Get damask load.yaml with given loadsteps. Default is uniaxial tension.

		Args:
			loadsteps: A list of load steps.
		"""
		# >------------------------------------------------------
		# default tension
		default_mechanical = {
			'dot_F': [[self.prm.rate, 0, 0],
					  [0, 'x', 0], 
					  [0, 0, 'x']],
			'P': [['x', 'x', 'x'],
				  ['x', 0, 'x'],
				  ['x', 'x', 0]]
		}
		loadstep1 = get_loadstep(default_mechanical,t=10,N=160,f_out=16)
		loadstep2 = get_loadstep(default_mechanical,t=60,N=120,f_out=2)
		default_loadsteps = [loadstep1, loadstep2]
		# >------------------------------------------------------

		loadsteps = loadsteps or default_loadsteps
		load = damask.Config(
			solver={'mechanical':'spectral_basic'},
			loadstep=loadsteps
		)
		load.save(f'{self.prm.WORKdir}/load.yaml')

	def getNumerics(self, integrator='RK4'):
		"""
		Get damask numerics.yaml with given integrator.

		Args:
			integrator: damask integrator, can be `FPI`, `RK4`, ...etc.
		"""
		num = damask.Config(
			crystallite={'integrator': integrator}
		)
		num.save(f'{self.prm.WORKdir}/numerics.yaml')

	def getZinfandel(self, n_proc=20):
		"""
		Get Zinfandel job.sh with given configuration.

		Args:
			n_proc: number of cpu processor.
		"""
		load = 'load' 
		grid = self.prm.TASKname
		taskname = self.prm.TASKname.split('_')[-1]
		job = (
		f'#!/bin/bash \n'
		f'#SBATCH --job-name="{self.prm.cond}-{self.prm.T}-{taskname}"\n'
		f'#SBATCH --partition=cpu-3g\n'
		f'#SBATCH --ntasks={n_proc}\n'
		f'#SBATCH --nodes=1-1\n'
		f'#SBATCH --output=cout.txt\n'
		f'#SBATCH --error=cerr.txt\n'
		f'#SBATCH --chdir=.\n'
		f'\n'
		f'sbatch_pre.sh\n'
		f'module load mpi\n'
		f'\n'
		f'load={load}\n'
		f'grid={grid}\n'
		f'OMP_NUM_THREADS=${{SLURM_NTASKS}} DAMASK_grid --load ${{load}}.yaml --geom ${{grid}}.vti > $(pwd)/${{grid}}.log\n'
		f'\n'
		f'sbatch_post.sh\n'
		)
		with open(f'{self.prm.WORKdir}/job.sh', 'w', newline='\n') as f:
			f.write(job)

	def preprocessing(self):
		"""
		Prepare all of the data to run Damask.
		"""
		if not os.path.exists(self.prm.WORKdir):
			os.mkdir(self.prm.WORKdir)
		self.getConfigMaterial()
		self.getGrid()
		self.getLoad()
		self.getZinfandel()

	# >----------------------------------------------------------------------------------------------------
	# > generating training data for surrogate model
	# >----------------------------------------------------------------------------------------------------
	def createPipeline(self, pipeline, texture, n_start, n_end):
		"""
		Create dream3d pipeline json file with given texture setting.

		Args:
			pipeline: dream3d pipeline.json
			texture: can be 'singleCrystal' / 'biCrystal' / 'random' / 'rolled'
			n_start: create pipeline from n_start
			n_end: create pipeline to n_end
		
		Creates:
			/ROOTdir
				|- texture_n
					|- texture_n.json
		"""

		odf_weights = pipeline["0"]["StatsDataArray"]["1"]["ODF-Weights"]
		num_grain = int(pipeline["1"]["EstimatedPrimaryFeatures"])
		
		for i, n in enumerate(range(n_start, n_end)):
			path = f'{self.prm.ROOTdir}/{texture}_{n}'
			if not os.path.exists(path):
				os.mkdir(path)

			if os.path.exists(f'{path}/{texture}_{n}.json'):
				print(f'<<< {texture}_{n}.json exists, createPipeline passed >>>')
				continue 

			e1, e2, e3, sigma, weight = get_weight(texture, num_grain)

			if 'morient' in texture:
				sigma[i] = int(texture.split('_')[-1])
				weight[i] = 500000

			odf_weights["Euler 1"] = e1.tolist()
			odf_weights["Euler 2"] = e2.tolist()
			odf_weights["Euler 3"] = e3.tolist()
			odf_weights["Sigma"] = sigma.tolist()
			odf_weights["Weight"] = weight.tolist()

			pipeline["0"]["StatsDataArray"]["1"]["ODF-Weights"] = odf_weights
			pipeline["5"]["OutputFile"] = f'{path}/{texture}_{n}.dream3d'
			pipeline["6"]["FeatureDataFile"] = f'{path}/{texture}_{n}.csv'

			with open(f'{path}/{texture}_{n}.json', 'w') as f:
				json.dump(pipeline, f, indent=4, separators=(',', ':'))
	
	def runPipeline(self, pRunner, texture, n_start, n_end):
		"""
		Run dream3d pipeline and prepare damask input files.

		Args:
			pRunner: path to pipeline runner.
			texture: type of texture, see `get_weight`.
			n_start: start index.
			n_end: end index.

		Creates:
			/ROOTdir
				|- texture_n
					|- texture_n.dream3d
					|- texture_n.csv
					|- material.yaml
					|- load.yaml
					|- geom.vti
					|- numerics.yaml
					|- job.sh
		"""

		for n in range(n_start, n_end):
			path = f'{self.prm.ROOTdir}/{texture}_{n}'

			iteration = 0
			print('-'*50)
			print(f'<<< running pipeline {texture}_{n}-{iteration} >>>')
			print('-'*50)

			if os.path.exists(f'{path}/{texture}_{n}.dream3d'):
				print(f'<<< {texture}_{n}.dream3d exists runPipeline passed >>>')
			else:
				# pipeline runner -> .dream3d
				args = f'{pRunner} -p {path}/{texture}_{n}.json'
				p = subprocess.Popen(args, shell=True)
				p.wait()
				iteration += 1

			# damask preprocessing -> material.yaml, load.yaml, grid.vti
			self.update_params()
			self.c_params()
			self.prm.WORKdir = f'{self.prm.ROOTdir}/{texture}_{n}'
			self.prm.RVEdir = f'{self.prm.ROOTdir}/{texture}_{n}'
			self.getConfigMaterial(dream3d=f'{texture}_{n}.dream3d')
			self.getGrid(dream3d=f'{texture}_{n}.dream3d')
			self.getLoad(loadsteps=get_loadsteps_RW(save_root=self.prm.WORKdir))
			self.getNumerics()
			self.getZinfandel()		

	# >----------------------------------------------------------------------------------------------------	
	def createGraphDataset(self):
		"""
		Create graph dataset from DAMASK.

		Creates:
			/raw
				|-texture_n
					|-data.pt
		"""
		texture_n = next(os.walk(self.prm.ROOTdir))[1]
		for n in texture_n:
			print('-'*50)
			print(f'<<< creating dataset {n} >>>')

			path = f'{self.prm.ROOTdir}/{n}'
			result_path = glob.glob(f'{path}/*.hdf5')
			grid_path = glob.glob(f'{path}/*.dream3d')
			csv_path = glob.glob(f'{path}/*.csv')

			if any([len(grid_path)==0, len(result_path)==0, len(csv_path)==0]):
				print(f'<<< passed: missing .dream3d or .hdf5 or .csv >>>')
				continue

			save_path = f'{self.prm.ROOTdir}/../raw'
			if not os.path.exists(f'{save_path}/{n}'):
				os.mkdir(f'{save_path}/{n}')
			else:
				print('<<< WARNING: dataset exists >>>')

			grid = damask.Grid.load_DREAM3D(grid_path[0], feature_IDs='FeatureIds') 
			result = damask.Result(result_path[0]).view(increments=list(range(0, 1051, 10)))

			eulers, num_neighbor, volume = get_node_features(grid, result, csv_path[0])
			edge_index, surface_areas, normals = get_edge_features(grid, csv_path[0])

			from_orient = damask.Orientation.from_Euler_angles(phi=eulers[0][edge_index[:,0]], family='cubic', lattice='cF')
			to_orient = damask.Orientation.from_Euler_angles(phi=eulers[0][edge_index[:,1]], family='cubic', lattice='cF')
			morients = from_orient.disorientation(to_orient).as_axis_angle()[..., -1:]			# (num_edges, 1)
			edge_attr = np.concatenate([normals, morients, surface_areas], axis=1)

			data = Data(
				euler=torch.tensor(eulers).float(),					# (seq_len, num_nodes, 3)
				num_neighbor=torch.tensor(num_neighbor).float(),	# (num_nodes, 1)
				volume=torch.tensor(volume).float(),				# (num_nodes, 1)
				edge_index=torch.tensor(edge_index).t().long(),		# (2, num_edges)
				edge_attr=torch.tensor(edge_attr).float(),			# (num_edges, 5)
				desc=n
			)
			torch.save(data, f'{save_path}/{n}/data.pt')

	# >----------------------------------------------------------------------------------------------------
	# > postprocessing
	# >----------------------------------------------------------------------------------------------------
	def getDAMASK3(self, target=None, func=lambda x: np.mean(x[:,0,0])):
		"""
		Get the stress-strain curve from the damask output.

		Args:
			target: The name of the quantity to extract.
			func: A function to process the extracted quantity from the damask result object in each increment.

		Returns:
			The engineering strain, engineering stress, yield point x, and yield point y.
		"""
		result = damask.Result(f'{self.prm.WORKdir}/{self.prm.TASKname}_load.hdf5')

		# get target
		if target:
			if result.get(target) is None:
				raise KeyError(f'cannot find {target} in hdf5!')
			return np.array([func(val) for val in result.get(target).values()])

		# get stress-strain
		if result.get('F') is None:
			result.add_calculation(
				'np.einsum("...ij, ...jk -> ...ik", #F_e#, #F_p#)', 
				'F', 'n/a', 'deformation gradient from F=FeFp')
		if result.get('epsilon_V^0.0(F)') is None:
			result.add_strain()
		if result.get('sigma') is None:
			result.add_stress_Cauchy()

		strain = np.array([func(e) for e in result.get('epsilon_V^0.0(F)').values()])
		stress = np.array([func(s) for s in result.get('sigma').values()])/1e6			# MPa

		E = 69000 																		# MPa, TODO: get E from actual slope
		strain, stress = get_engineering(strain, stress)
		YSx, YSy = get_YS(strain, stress, E)

		return strain, stress, YSx, YSy

