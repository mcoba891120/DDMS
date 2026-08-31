import os
import damask
import pandas as pd
import glob
import logging
from . import constitutive
from .processing import get_YS

logger = logging.getLogger(__name__)

# root directory holding experimental/RVE input data (TEM csv, stress-strain
# xlsx, RVE meshes, ...); override via the DDMS_DATA_ROOT env var so scripts
# don't have to be run from a fixed working directory
DATA_ROOT = os.environ.get('DDMS_DATA_ROOT', '../../data')

class Handler():
	def __init__(self, YAML, taskname, target, ROOTdir):
		logger.info('loading parameters from %s', YAML)
		self.template, self.prm = self.loadConfigMaterialTemplate(YAML, return_mat=True)
		self.prm.taskname = taskname 
		self.prm.target = target  
		self.prm.ROOTdir = ROOTdir

	def loadConfigMaterialTemplate(self, YAML, return_mat=False):
		"""
		Load the damask config material template and return the material parameters.

		Args:
			YAML: The YAML file path of the material template.
			return_mat: Whether to return the damask config material object.

		Returns:
			The material parameters or the damask config material object and the material parameters.
		"""
		mat = damask.ConfigMaterial.load(YAML)
		prm = {
			**mat['system'],
			**mat['phase']['Aluminum']['mechanical']['plastic']
		}
		prm = type('CONFIG', (), prm)
		if return_mat:
			return mat, prm
		else:
			return prm

	def update_params(self, taskname='', target='', ROOTdir='', verbose=1):
		"""
		Update system parameters for each temperatures (Ts) and aging conditions (conds).

		Args:
			taskname: The name of the task.
			target: The name of the quantity to update.
			ROOTdir: The root directory of the project.
			verbose: The verbosity level.
		"""
		next(self.params_updater(taskname, target, ROOTdir, verbose))

	def params_updater(self, taskname='', target='', ROOTdir='', verbose=1):
		"""
		Wrapper function of `_params_updater`. 

		Args:
			taskname: The name of the task.
			target: The target aging condition.
			ROOTdir: The root directory of the project.
			verbose: Whether to print the task information.

		Returns:
			The `_params_updater` generator
		"""
		taskname = taskname or self.prm.taskname
		target   = target or self.prm.target
		ROOTdir  = ROOTdir or self.prm.ROOTdir
		return self._params_updater(taskname, target, ROOTdir, verbose)
	
	def c_params(self):
		""" Calculate all parameters """
		self.c_TEM()
		self.c_YS()
		self.c_PK()

	def g_params(self):
		""" Report parameters information """
		self.g_TEM()
		self.g_YS()
		self.g_WH()

	def getEXP(self):
		"""
		Get the experimental stress-strain curve.

		Returns:
			The engineering strain, engineering stress, yield point x, and yield point y.
		"""
		data = pd.read_excel(f'{DATA_ROOT}/exp/{self.prm.alloy}/SScurve/{self.prm.alloy}_new.xlsx')
		idx = data.columns.get_loc(f'{self.prm.cond}_{self.prm.T}K')
		strain = data.iloc[:,idx+1].dropna().values/100																	# _
		stress = data.iloc[:,idx+2].dropna().values																		# MPa
		E = 69000 																										# MPa, TODO: get E from actual slope

		# data having incorrect Youngs module
		if data.iloc[0,idx]=='correction':
			fake_E = stress[1]/strain[1] 																				# MPa

			if self.prm.cond=='7min' and self.prm.T==473:
				fake_E *= 0.97																							# broken Young's module, fix it manually

			fake_YSx, YSy = get_YS(strain, stress, E=fake_E)
			YSx = YSy/E+0.002
			
		# data having only plastic strain
		elif data.iloc[0,idx]=='plastic':
			YSx = strain[0]
			YSy = stress[0]

		else:
			YSx, YSy = get_YS(strain, stress, E=E)

		return strain, stress, YSx, YSy


	# >----------------------------------------------------------------------------------------------------
	def _params_updater(self, taskname, target, ROOTdir, verbose=1):
		"""
		Update system parameters for each temperatures (Ts) and aging conditions (conds).

		Args:
			taskname: The name of the task.
			target: The target aging condition.
			ROOTdir: The root directory of the project.
			verbose: Whether to print the task information.

		Yields:
			The updated parameters.
		"""
		for T in self.prm.Ts: 
			self.prm.T = T
			for cond, rl in zip(self.prm.conds, self.prm.rls):
				
				self.prm.YS_exp = self.prm.YS_exps.get(f'{cond}_{T}K', None)
				if not self.prm.YS_exp:
					continue

				if not target in f'{cond}_{T}K' and not len(target)==0: 
					continue

				self.prm.cond = cond
				self.prm.rl = rl
				self.prm.TASKname = f'{self.prm.alloy}_{cond}_{T}K_{taskname}'
				self.prm.WORKdir = f'{ROOTdir}/{self.prm.TASKname}'
				self.prm.RVEdir = f'{DATA_ROOT}/rve/{self.prm.alloy}/RVE/{cond}'
				self.prm.PBCname = f'{DATA_ROOT}/rve/poly8000'
				self.prm.EXPname = glob.glob(f'{DATA_ROOT}/exp/{self.prm.alloy}/TEM/{cond}.csv')
				self.prm.EXPname = self.prm.EXPname[0] if self.prm.EXPname else None
				self.prm.EXPnames = glob.glob(f'{DATA_ROOT}/exp/{self.prm.alloy}/TEM/{cond}/*.csv') 	# 7075

				if verbose:
					logger.info('#' * 70)
					logger.info('TASK: %s', self.prm.TASKname)
					logger.info('RVE: %s', self.prm.RVEdir)
					logger.info('PBC: %s', self.prm.PBCname)
				yield

	def c_TEM(self):
		if self.prm.c_TEM[0]=='':
			logger.info('c_TEM passed (no functions configured)')
			return
		for funcs in self.prm.c_TEM:
			self.prm = getattr(constitutive, funcs)(self.prm)

	def c_YS(self):
		if self.prm.c_YS[0]=='':
			logger.info('c_YS passed (no functions configured)')
			return
		for funcs in self.prm.c_YS:
			self.prm = getattr(constitutive, funcs)(self.prm)

	def c_PK(self):
		if not self.prm.c_PK:
			logger.info('c_PK passed (no functions configured)')
			return
		for funcs in self.prm.c_PK:
			self.prm = getattr(constitutive, funcs)(self.prm)
		self.c_YS()

	def g_TEM(self):
		logger.info("-"*50)
		logger.info('%s', f'{"TEM":^50s}')
		logger.info("-"*50)
		lmb = '\u03BB'
		try:
			if not self.prm.c_PK:
				logger.info(f'{"R_mean":10s}: {self.prm.R_mean*1e9:10.2f} nm')
				logger.info(f'{"Req_mean":10s}: {self.prm.Req_mean*1e9:10.2f} nm')
				logger.info(f'{"R_std":10s}: {self.prm.R_std*1e9:10.2f} nm')
				logger.info(f'{"Nv":10s}: {self.prm.Nv/1e20:10.2f} m^-3')
				logger.info(f'{"Nvo":10s}: {self.prm.Nvo/1e20:10.2f} m^-3')
				logger.info(f'{"Nv_std":10s}: {self.prm.Nv_std/1e20:10.2f} m^-3')
				logger.info(f'{"f_all":10s}: {self.prm.f_all:10.2%}')
				logger.info(f'{"f_o":10s}: {self.prm.f_o:10.2%}')
				logger.info(f'{"phi":10s}: {self.prm.phi:10.2f}')
				logger.info(f'{lmb:10s}: {self.prm.lmb*1e9:10.2f} nm')
				logger.info(f'{lmb+"*":10s}: {self.prm.lmb_star*1e9:10.2f} nm')
			else:
				logger.info(f'{"R_mean":10s}: {self.prm.R_mean*1e9:10.2f} nm')
				logger.info(f'{"Req_mean":10s}: {self.prm._Req_mean*1e9:10.2f} -> {self.prm.Req_mean*1e9:10.2f} nm')
				logger.info(f'{"R_std":10s}: {self.prm.R_std*1e9:10.2f} nm')
				logger.info(f'{"Nv":10s}: {self.prm._Nv/1e20:10.2f} -> {self.prm.Nv/1e20:10.2f} m^-3')
				logger.info(f'{"Nvo":10s}: {self.prm._Nvo/1e20:10.2f} -> {self.prm.Nvo/1e20:10.2f} m^-3')
				logger.info(f'{"Nv_std":10s}: {self.prm.Nv_std/1e20:10.2f} m^-3')
				logger.info(f'{"f_all":10s}: {self.prm._f_all:10.2%} -> {self.prm.f_all:10.2%}')
				logger.info(f'{"f_o":10s}: {self.prm._f_o:10.2%} -> {self.prm.f_o:10.2%}')
				logger.info(f'{"phi":10s}: {self.prm._phi:10.2f} -> {self.prm.phi:10.2f}')
				logger.info(f'{lmb:10s}: {self.prm._lmb*1e9:10.2f} -> {self.prm.lmb*1e9:10.2f} nm')
				logger.info(f'{lmb+"*":10s}: {self.prm._lmb_star*1e9:10.2f} -> {self.prm.lmb_star*1e9:10.2f} nm')
		except AttributeError:
			# expected when c_TEM() hasn't populated these fields yet
			logger.info('TEM passed (parameters not computed)')

	def g_YS(self):
		logger.info("-"*50)
		logger.info('%s', f'{"Yield Stress model":^50s}')
		logger.info("-"*50)
		try:
			if not self.prm.c_PK:
				logger.info(f'\u03C3gb   : {self.prm.gb*1e-6:10.2f} MPa')
				logger.info(f'\u03C3ss   : {sum(self.prm.ss)*1e-6:10.2f} MPa')
				logger.info(f'\u03C3p    : {self.prm.pp*1e-6:10.2f} MPa')
				logger.info(f'\u03C3y    : {self.prm.YS*1e-6:10.2f} MPa')
				logger.info(f'\u03C3y err: {self.prm.YS*1e-6-self.prm.YS_exp:10.2f} MPa')
			else:
				logger.info(f'\u03C3gb   : {self.prm.gb*1e-6:10.2f} MPa')
				logger.info(f'\u03C3ss   : {sum(self.prm._ss)*1e-6:10.2f} -> {sum(self.prm.ss)*1e-6:10.2f} MPa')
				logger.info(f'\u03C3p    : {self.prm._pp*1e-6:10.2f} -> {self.prm.pp*1e-6:10.2f} MPa')
				logger.info(f'\u03C3y    : {self.prm._YS*1e-6:10.2f} -> {self.prm.YS*1e-6:10.2f} MPa')
				logger.info(f'\u03C3y err: {self.prm._YS*1e-6-self.prm.YS_exp:10.2f} -> {self.prm.YS*1e-6-self.prm.YS_exp:10.2f} MPa')
				logger.info(f'\u03C3y err: {abs(self.prm._YS*1e-6-self.prm.YS_exp)/self.prm.YS_exp:10.2%} -> {abs(self.prm.YS*1e-6-self.prm.YS_exp)/self.prm.YS_exp:10.2%}')
		except AttributeError:
			# expected when c_YS()/c_PK() hasn't populated these fields yet
			logger.info('YS passed (parameters not computed)')

	def g_WH(self):
		logger.info("-"*50)
		logger.info('%s', f'{"Work Hardening model":^50s}')
		logger.info("-"*50)
		try:
			logger.info(f'{"k2":10s}: {self.prm.k2:10.2f}')
			logger.info(f'{"k2g":10s}: {self.prm.k2g:10.2f}')
		except AttributeError:
			# expected when the work-hardening model hasn't been run yet
			logger.info('WH passed (parameters not computed)')

	

