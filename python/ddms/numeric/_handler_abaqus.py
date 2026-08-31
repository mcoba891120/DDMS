from ._handler import Handler 
from datetime import datetime
from subprocess import call, PIPE, Popen
from intersect import intersection
import os 
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('tkAgg')


class HandlerAbaqus(Handler):
	def __init__(self, YAML, taskname, target, ROOTdir):
		super().__init__(YAML, taskname, target, ROOTdir)

	# >--------------------------------------------------------------
	# > preprocessing
	# >--------------------------------------------------------------
	def writeINP(self, deform=9):
		with open(f'{self.prm.RVEdir}/RVE_all.inp','r')as f:
			geoStrs = f.readlines()
			DEL = END = len(geoStrs)
			for i, line in enumerate(geoStrs):
				if ' 999999,' in line:
					DEL = i
				if '*End Instance' in line:
					END = i+2

		find = '*Material, name=GRAIN_MAT1'
		find2 = '** BOUNDARY CONDITIONS'
		BCstart = []
		with open(f'{self.prm.PBCname}.inp','r') as f:
			BCstrs = f.readlines()
			for i, line in enumerate(BCstrs):
				if find in line:
					MATstart = i
				if find2 in line:
					BCstart.append(i)

		with open(f'{self.prm.WORKdir}/{self.prm.TASKname}.inp', 'w')as inp:
			# write util the line before --*End Assembly--, and delete node 999999
			for i in range(END):
				if i == DEL:
					continue
				else:
					inp.write(geoStrs[i])

			# write util --*End Assembly--
			for i in range(MATstart):
				inp.write(BCstrs[i])
			
			# write --MATERIALS property--

			# Read initial orientation data from csv file
			oris = pd.read_csv(f'{self.prm.RVEdir}/RVE.csv',header=1)
			ori = oris.loc[:,['EulerAngles_0','EulerAngles_1','EulerAngles_2']].values  
			for i, eu in enumerate(ori):	
				s1, s0, s2 = np.sin(eu[0]), np.sin(eu[1]), np.sin(eu[2])
				c1, c0, c2 = np.cos(eu[0]), np.cos(eu[1]), np.cos(eu[2])
			
				R = np.stack([[c1*c2-s1*s2*c0, s1*c2+c1*s2*c0, s2*s0],
							  [-c1*s2-s1*c2*c0, -s1*s2+c1*c2*c0, c2*s0],
							  [s1*s0, -c1*s0, c0]],axis=1)

				inp.write(f'*Material, name=GRAIN_MAT{i+1}\n')

				for j in range(5):
					inp.write(BCstrs[MATstart+1+j])

				n_sl 		= self.prm.n_sl 
				rho_0 		= self.prm.rho_ssd_0[0]
				tau 		= self.prm.tau 
				alpha 		= self.prm.alpha
				G 			= self.prm.G 
				b 			= self.prm.b 
				k1 			= self.prm.k1 
				k2 			= self.prm.k2 
				L 			= self.prm.L 
				sat 		= self.prm.sat 
				a_gnd 		= self.prm.a 
				h_latent 	= 1.4
				h_self 		= 1.0

				props = [
						[1.      , 0.    , 0.   , R[0][0], R[1][0], R[2][0], 0., 0.],   	# PROPS(17)-PROPS(32), global index
						[0.      , 0.    , 1.   , R[0][2], R[1][2], R[2][2], 0., 0.],
						[n_sl    , 0.001 , 0.   , 0.     , 0.     , 0.     , 0., rho_0],	# PROPS(33)-PROPS(40), global index
						[tau     , alpha , G    , b      , k1     , k2     , L , sat], 		# PROP(1)-PROP(8), HSELF index, ssd_v1 + gnd_v2
						[h_latent, h_self, a_gnd, 0.     , 0.     , 0.     , 0., 0.] 		# PROP(9)-PROP(16), HSELF index, ssd_v1 + gnd_v2
						]
				for prs in props:
					formatted_prs = list(map(lambda x: f'{x:+10.4e}', prs))
					inp.write(','.join(formatted_prs)+'\n')

				# inp.write(f'1.,0.,0.,{float(R[0][0]):10.6f},{float(R[1][0]):10.6f},{float(R[2][0]):10.6f},0.,0.\n')
				# inp.write(f'0.,0.,1.,{float(R[0][2]):10.6f},{float(R[1][2]):10.6f},{float(R[2][2]):10.6f},0.,0.\n')
				# inp.write(f'{self.prm.m:9d}.,0.001,0.,0.,0.,0.,0.,1.e-5\n')
				# # PROP(1) - PROP(16) in HSELF
				# inp.write(f'{float(self.prm.tau):10.1f},       0.3,{float(self.prm.G):10.1f},   2.86e-7,{float(self.prm.k1):10.1f},{float(self.prm.k2):10.2f},{self.prm.L:10.2e},{self.prm.sat:10.2e}\n')
				# # ssd_v1 + gnd_v1
				# inp.write(f'       1.4,        1., {self.prm.a:10.3f},	  0.,		 0.,		0.,		   0., 		  0.\n')
				# # ssd_v1 + gnd_v2
				# inp.write(f'       1.4,        1., {self.prm.a:10.3f},{self.prm.k1g/self.prm.lamda:10.1f},{self.prm.k2g:10.1f},		0.,		   0., 		  0.\n')
				# # ssd_v2 + gnd_v1
				# inp.write(f'       1.4,        1., {self.prm.a:10.3f},{self.prm.L_ssd:.3f},{self.prm.phi:.3f},{self.prm.k20:.3f},{self.prm.k2p:.3f}, 0.\n')
					
			for i in range(BCstart[0]-1,BCstart[1]-2):
				inp.write(BCstrs[i])
			
			inc = deform*100
			inp.write(f'{float(0.00001)}, {int(deform/self.prm.rate)}., {1e-5}, {float(deform/self.prm.rate/inc)}\n')
			
			for i in range(BCstart[1]-1,BCstart[1]+4):
				inp.write(BCstrs[i])
			inp.write(f'RightLeftPair, 1,1, {float(self.prm.rate)}\n')
			for i in range(BCstart[1]+5,len(BCstrs)):
				inp.write(BCstrs[i])
					
		print(f'<<< write {self.prm.TASKname}.inp >>>')

	def getZinfandel(self):
		taskname = self.prm.TASKname.split('_')[-1]
		job = (
		f'#!/bin/bash \n'
		f'#SBATCH --job-name="{self.prm.cond}-{self.prm.T}-{taskname}"\n'
		f'#SBATCH --partition=96-cores\n'
		f'#SBATCH --ntasks=16\n'
		f'#SBATCH --output=cout.txt\n'
		f'#SBATCH --error=cerr.txt\n'
		f'#SBATCH --chdir=.\n'
		f'\n'
		f'module load opt gcc compiler/2021.2.0 abaqus/2019-2\n'
		f'sbatch_pre.sh\n'
		f'\n'
		f'unset SLURM_GTIDS\n'
		f'\n'
		f'abaqus input={self.prm.TASKname}.inp job={self.prm.TASKname} user=UMAT_ssd1_gnd2.for cpus=$SLURM_NTASKS mp_mode=thread scratch=. interactive\n'
		f'\n'
		f'abaqus viewer noGUI=output.py\n'
		f'\n'
		f'sbatch_post.sh\n'
		)
		with open(f'{self.prm.WORKdir}/job.sh', 'w', newline='\n') as f:
			f.write(job)

	def getOutput(self):
		with open('./core/abaqus/output.py','r')as f:
			tempStrs = f.readlines()
			
		with open(f'{self.prm.WORKdir}/output.py','w')as f:
			for i in range(len(tempStrs)):
				if i == 5:
					f.write('TASKname = \''+str(self.prm.TASKname)+'\'\n')
				else:
					f.write(tempStrs[i])

	def getUMAT(self):
		shutil.copy('./core/abaqus/UMAT_ssd1_gnd2.for', f'./{self.prm.WORKdir}/UMAT_ssd1_gnd2.for')

	def runjob(self):
		S = datetime.now()
		print(f'')
		print(f'Job \'{self.prm.TASKname}\' starts at: {S}\n')
		
		process = call("@echo off", shell=True, stdout=PIPE)
		process = Popen(f'call ABAQUS cpus=4 input={self.prm.TASKname}.inp job={self.prm.TASKname} user=./core/UMAT_ssd1_gnd2.for int', shell=True, stdout=PIPE)
		process.wait()
		# print(process.stdout.read())

		E = datetime.now()
		print(f'Job \'{self.prm.TASKname}\' ends at: {E}')
		print(f'Job duration: {E-S}\n') 

	def preprocessing(self):
		if not os.path.exists(self.prm.WORKdir):
			os.mkdir(self.prm.WORKdir)
		self.writeINP()
		self.getZinfandel()
		self.getOutput()
		self.getUMAT()

	# >--------------------------------------------------------------
	# > postprocessing
	# >--------------------------------------------------------------
	def getABAQUS(self, rho=False):
		if rho:
			data = pd.read_csv(f'{self.prm.WORKdir}/Rho_{self.prm.TASKname}.txt', sep='\s+')
			rhos = data.iloc[:,0].values
			rhog = data.iloc[:,1].values
			return rhos, rhog

		data = pd.read_csv(f'{self.prm.WORKdir}/StressStrain_{self.prm.TASKname}.txt', sep='\s+')
		strain = data.iloc[:,0].values
		stress = data.iloc[:,1].values/1e6
		eng_strain = np.exp(strain)-1
		eng_stress = stress/(1+eng_strain)

		ylim = max(eng_stress)
		E = 69000 	# MPa, TO UPDATE: get E from actual slope
		# E = np.mean(
		# 	[stress/strain for strain, stress in zip(eng_strain,eng_stress) if strain<=0.002 and strain!=0]
		# 	)
		YSx, YSy = intersection(eng_strain, eng_stress,
								np.array([0.002, ylim/E+0.002]),
								np.array([0,ylim]))
		return eng_strain, eng_stress, YSx, YSy

	def plotABAQUS(self):
		exp_strain, exp_stress, exp_YSx, exp_YSy = self.getEXP()
		eng_strain, eng_stress, YSx, YSy = self.getABAQUS()

		plt.rc('font', size=25)
		_, ax1 = plt.subplots(figsize=(16,10))
		plt.grid()

		ax1.plot(exp_strain, exp_stress, 'kx', alpha=0.6, markersize=3)
		ax1.plot(eng_strain, eng_stress, c='cyan', alpha=0.6, linewidth=3)
		if exp_YSy[0]: ax1.scatter(exp_YSx[0], exp_YSy[0], c='k', label=f'{exp_YSy[0]:.2f}MPa, EXP')
		if YSy[0]: ax1.scatter(YSx[0], YSy[0], c='cyan', label=f'{YSy[0]:.2f}MPa, ABAQUS') 

		ax1.set_xlabel('strain (_)')
		ax1.set_ylabel('stress (MPa)')
		ax1.set_ylim(0,400)
		ax1.set_title(self.prm.TASKname, y=1.02)
		ax1.legend()
		plt.savefig(f'{self.prm.WORKdir}/{self.prm.TASKname}.png')

	def postprocessing(self):
		self.plotABAQUS()




