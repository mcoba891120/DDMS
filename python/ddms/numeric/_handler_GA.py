"""
TODO: DEPRECATED
"""

from ._handler_damask import HandlerDamask 
from tqdm import tqdm
import numpy as np
import pandas as pd  
import os
import matplotlib.pyplot as plt   

class HandlerGA(HandlerDamask):
	def __init__(self, YAML, taskname, target, ROOTdir):
		super().__init__(YAML, taskname, target, ROOTdir)

		# -----------------------------------------
		"""assuming 5 levels, i.e. [-2,-1,0,1,2]"""
		self.IsYS = False
		self.tname = np.array(['k1','k1g','Zs','Zg'])    			# thetas name
		self.tl2 = np.array([4.0e+8,4.0e+8,1.0e+5,1.0e+8])/1.25 	# thetas lower2
		self.tu2 = np.array([4.0e+8,4.0e+8,1.0e+5,1.0e+8])*1.25 	# thetas upper2
		# -----------------------------------------

		# # >-----------------------------------------
		# """assuming 5 levels, i.e. [-2,-1,0,1,2]"""
		# self.IsYS = True
		# self.tname = np.array(['beta','M'])  	# thetas name
		# self.tl2 = np.array([0.1, 2.0]) 		# thetas lower2
		# self.tu2 = np.array([0.5, 3.0]) 			# thetas upper2
		# # >-----------------------------------------

	# >--------------------------------------------------------------
	# > preprocessing
	# >--------------------------------------------------------------
	def preRSM(self):
		thetas_center = (self.tu2 + self.tl2)/2
		thetas_lower1 = (self.tl2 + thetas_center)/2
		thetas_upper1 = (thetas_center + self.tu2)/2
		
		self.thetas = np.array([])
		# corner trials
		def generateCornerTrials(ti=0, thetas_tmp=np.array([])):
			if len(thetas_tmp)>=len(self.tname):
				self.thetas = np.append(self.thetas, thetas_tmp)
				return 

			for t in [thetas_lower1[ti], thetas_upper1[ti]]:
				generateCornerTrials(ti+1, np.append(thetas_tmp, t))

		# axis trials
		def generateAxisTrials():
			for ti in range(len(self.tname)):
				for t in [self.tl2[ti], self.tu2[ti]]:
					thetas_tmp = thetas_center
					thetas_tmp[ti] = t
					self.thetas = np.append(self.thetas, thetas_tmp)
		# center trials
		def generateCenterTrials(n=1):
			for _ in range(n):
				self.thetas = np.append(self.thetas, thetas_center)

		generateCornerTrials()
		generateAxisTrials()
		generateCenterTrials()
		self.thetas = self.thetas.reshape(-1,len(self.tname))
		for i, theta in enumerate(self.thetas):
			for _ in self.params_updater(taskname=f'{self.prm.taskname}{i}'):
				self.c_params()
				self.g_params()
				for ti, t in enumerate(theta):
					setattr(self.prm, self.tname[ti], float(t))
				self.preprocessing()


	def c_RSMinit(self, strain_discret=9) -> list:
		tns = np.unique(
			[TASKname.split('_')[-1] for TASKname in os.listdir(self.prm.ROOTdir)]
		)

		xs = np.array([])  	# normalized params matrix
		ys = np.array([]) 	# simulation results
		for tn in tns:
			for _ in self.params_updater(taskname=tn, verbose=0):
				prm = self.loadConfigMaterial(f'{self.prm.WORKdir}/material.yaml')
				strain, stress, YSx, YSy = self.getDAMASK3()

				xs = np.append(xs, self.x_15(getattr(prm, self.tname[0]), 
										     getattr(prm, self.tname[1]), 
										     getattr(prm, self.tname[2]), 
										     getattr(prm, self.tname[3])))
				ys = np.append(ys, self.sigma_9(strain, stress, YSx, YSy))

		xs = xs.reshape(-1, len(self.prm.YS_exps), 15).swapaxes(0,1) 				# shape of xs (YS_exps, thetas_comb, 15)
		ys = ys.reshape(-1, len(self.prm.YS_exps), 9).swapaxes(0,1).swapaxes(1,2)   # shape of ys (YS_exps, thetas_comb, 9)
		# xs = xs.reshape(-1, 4, 15).swapaxes(0,1) 				# shape of xs (YS_exps, thetas_comb, 15)
		# ys = ys.reshape(-1, 4, 9).swapaxes(0,1).swapaxes(1,2)   # shape of ys (YS_exps, 9, thetas_comb)
		
		self.RSMs = []
		for x, y in zip(xs, ys):
			x = np.matrix(x)
			y = np.matrix(y)
			b_9 = np.array([(x.T*x).I*x.T*y_25.T for y_25 in y]).reshape(9,15)
			self.RSMs.append([self.sigma_f(b) for b in b_9])

		# ################# plot region #################
		# Cprms = np.array([
		# 	self.TtoC(getattr(prm, self.tname[ti]), self.tl2[ti], self.tu2[ti]) 
		# 	for ti in range(len(self.tname))
		# ])

		# sigma_hats = np.array([
		# 	rsm(Cprms[0], Cprms[1], Cprms[2], Cprms[3]) 
		# 	for rsm in RSMs[-1]
		# ])

		# strain, stress, YSx, YSy = self.getDAMASK3()
		# print(sigma_hats)

		# import matplotlib.pyplot as plt
		# plt.plot(strain, stress, c='c', label='damask')
		# plt.plot(np.arange(0, 0.09, 0.01), sigma_hats, c='salmon', label='rsm')
		# plt.ylim(0, 350)
		# plt.xlabel('plastic strain')
		# plt.ylabel('stress (MPa)')
		# plt.show()
		# ################# plot region #################
		return self.RSMs 	# RSMs:list, shape==(conds, 9)

	def c_RSM(self):
		# update RSM at the end of a gen
		pass

	def c_GA(self):
		self.DNA_SIZE = 10					# DNA length
		self.POP_SIZE = 100				# population size
		self.CROSS_RATE = 0.3				# mating probability (DNA crossover)
		self.MUTATION_RATE = 0.003			# mutation probability
		self.N_GENERATIONS = 500			# assume the converge maximum
		if not self.IsYS:
			self.RSMs = self.c_RSMinit() 		# RSMs:list, shape==(conds, 9)
				
		# initialize
		pop = np.random.randint(2, size=(len(self.tname), self.POP_SIZE, self.DNA_SIZE))
		
		# GA start
		for _ in tqdm(range(self.N_GENERATIONS)):
			params = np.array([
				self.translateDNA(pop[ti],self.tl2[ti], self.tu2[ti]) 
				for ti in range(len(self.tname))
			])
			fitness = self.get_fitness(params)

			################# TO UPDATE #################
			percent = int(len(fitness)*0.7)
			b50_index = np.argsort(fitness)[:percent]
			b50_params = params[:, b50_index]
			b50_diff = np.max(b50_params, axis=1)-np.min(b50_params, axis=1)
			extend10 = (self.tu2-self.tl2)*0.1
			if np.all(b50_diff < extend10):
				break
			################# TO UPDATE #################
			pop = self.select(pop, fitness)
			pop = self.crossover_mutate(pop, pop.copy())

		best10_indices = np.argsort(fitness)[:10]
		best10_params = np.array([
			self.translateDNA(pop[ti][best10_indices],self.tl2[ti], self.tu2[ti]) 
			for ti in range(len(self.tname))
		])

		best10_df = pd.DataFrame.from_dict(
			{tn: best10_params[ti] for ti, tn in enumerate(self.tname)})
		best10_df['fitness'] = fitness[best10_indices]
		print(f'<<< best 10 parameters >>>')
		print(best10_df)

	# >--------------------------------------------------------------
	# > utility functions
	# >--------------------------------------------------------------
	def TtoC(self,Tnum, Tmin, Tmax, Cmin=-2, Cmax=2):
		# get normalized value
		if (Tmax-Tmin)==0:
			return 0
		return Cmax-(Cmax-Cmin)*(Tmax-Tnum)/(Tmax-Tmin)

	def x_15(self, *args) -> list:
		# get normalized params matrix
		# WARNING: assuming 4 thetas
		x = [1]
		x.append(self.TtoC(args[0],self.tl2[0],self.tu2[0]))
		x.append(self.TtoC(args[1],self.tl2[1],self.tu2[1]))
		x.append(self.TtoC(args[2],self.tl2[2],self.tu2[2]))
		x.append(self.TtoC(args[3],self.tl2[3],self.tu2[3]))
		x.append(self.TtoC(args[0],self.tl2[0],self.tu2[0])*self.TtoC(args[1],self.tl2[1],self.tu2[1]))
		x.append(self.TtoC(args[0],self.tl2[0],self.tu2[0])*self.TtoC(args[2],self.tl2[2],self.tu2[2]))
		x.append(self.TtoC(args[0],self.tl2[0],self.tu2[0])*self.TtoC(args[3],self.tl2[3],self.tu2[3]))
		x.append(self.TtoC(args[1],self.tl2[1],self.tu2[1])*self.TtoC(args[2],self.tl2[2],self.tu2[2]))
		x.append(self.TtoC(args[1],self.tl2[1],self.tu2[1])*self.TtoC(args[3],self.tl2[3],self.tu2[3]))
		x.append(self.TtoC(args[2],self.tl2[2],self.tu2[2])*self.TtoC(args[3],self.tl2[3],self.tu2[3]))
		x.append(self.TtoC(args[0],self.tl2[0],self.tu2[0])**2)
		x.append(self.TtoC(args[1],self.tl2[1],self.tu2[1])**2)
		x.append(self.TtoC(args[2],self.tl2[2],self.tu2[2])**2)
		x.append(self.TtoC(args[3],self.tl2[3],self.tu2[3])**2)
		return x

	def sigma_f(self,b):
		return lambda A,B,C,D: (
			b[0] + b[1]*A + b[2]*B + b[3]*C + b[4]*D +
			b[5]*A*B + b[6]*A*C + b[7]*A*D + b[8]*B*C + b[9]*B*D +
			b[10]*C*D + b[11]*A*A + b[12]*B*B + b[13]*C*C + b[14]*D*D
		)

	def translateDNA(self, pop, lower=0.0, upper=1.0):
		# convert binary DNA to decimal and normalize it to a range(a, b)
		extend = upper - lower
		return pop.dot(2**np.arange(self.DNA_SIZE)[::-1]) / float(2**self.DNA_SIZE-1) * extend + lower			

	def get_fitness(self,params):
		assert params.shape==(len(self.tname), self.POP_SIZE)

		fitness = np.zeros(self.POP_SIZE)
		for c, _ in enumerate(self.params_updater(verbose=0)):
			if self.IsYS:
				for ti, tname in enumerate(self.tname):
					setattr(self.prm, tname, params[ti])
				self.c_TEM()
				self.c_YS()
				sigma_hats = self.prm.YS/1e6
				sigma = self.prm.YS_exp
				fitness += np.sqrt(np.power(sigma_hats-sigma, 2)/np.power(sigma, 2))
				# fitness += np.abs(sigma_hats-sigma)
			else:	
				# Cprms.shape == params.shape
				Cprms = np.array([
					self.TtoC(params[ti], self.tl2[ti], self.tu2[ti]) 
					for ti in range(len(self.tname))
				])

				# sigma_hats.shape == (POP_SIZE, strain(=9))
				sigma_hats = np.array([
					rsm(Cprms[0], Cprms[1], Cprms[2], Cprms[3]) for rsm in self.RSMs[c]
				]).reshape(self.POP_SIZE, -1)

				strain, stress, YSx, YSy = self.getEXP(self.prm.TASKname)
				sigma = self.sigma_9(strain, stress, YSx, YSy)
				################# TO UPDATE #################
				d1 = np.sqrt(np.sum(np.power(sigma_hats-sigma, 2), axis=1) / \
							 np.sum(np.power(sigma, 2)))
				
				d2_hats = sigma_hats[:,1:]-sigma_hats[:,0].reshape(-1,1)
				d2_exp = sigma[1:]-sigma[0]
				d2 = np.sum(np.abs(d2_hats - d2_exp) / np.abs(d2_exp), axis=1)
				# fitness += np.sum(np.abs(sigma_hats-sigma), axis=1) 
				fitness += d1+d2
				# fitness += d1
				################# TO UPDATE #################		

		return fitness

	def select(self, pop, fitness):
		################# TO UPDATE #################
		fitness = np.abs(fitness - max(fitness))**50
		################# TO UPDATE #################
		idx = np.random.choice(
			np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True, p = fitness/fitness.sum()
		)
		pop[:] = pop[:, idx]
		return pop

	def crossover_mutate(self, pop, pop_copy):
		# pop.shape == (len(self.tname), POP_SIZE, DNA_SIZE)
		cross_pops_mask = np.random.choice(
			[False,True], size=pop.shape[:2], p=[1-self.CROSS_RATE, self.CROSS_RATE]
		) 																				# boolean mask for pop to be crossovered
		np.random.shuffle(pop_copy) 													# shuffle the pop_copy

		cross_points_mask = np.random.randint(
			0, 2, size=pop[cross_pops_mask].shape
		).astype(np.bool) 																# determine crossover points

		pop[cross_pops_mask] = np.where(
			cross_points_mask, pop_copy[cross_pops_mask], pop[cross_pops_mask]
		) 																				# crossover
		
		MUTATION_MASK = np.random.choice(
			[False,True], size=pop.shape, p=[1-self.MUTATION_RATE, self.MUTATION_RATE]
		)
		pop[MUTATION_MASK] = 1
		return pop


	# >--------------------------------------------------------------
	# > DEPRECATED
	# >--------------------------------------------------------------
	# def c_GA_YS(self):
	# 	best_params = np.array([])
	# 	for _ in range(10):
	# 		best_params = np.append(best_params, self.c_GA())

	# 	best_params = (best_params.reshape(-1,4)-self.tl2)/(self.tu2-self.tl2)
	# 	df = pd.DataFrame(best_params, columns=self.tname)
	# 	plt.figure(figsize=(16,10))
	# 	plt.rc('font', size=25)
	# 	f = df.boxplot(sym = 'o',
	#               vert = True,
	#               whis = 1.5,
	#               patch_artist = True,
	#               meanline = False,
	#               showmeans = True,
	#               showbox = True,
	#               showcaps = True,
	#               showfliers = True,
	#               notch = False, 
	#               return_type = 'dict'
	#               )
	# 	plt.ylim(0.5,1.0)
	# 	plt.show()




