import numpy as np
import matplotlib.pyplot as plt 
from .processing import get_plastic, get_uniform_stress, smoothing 


def plot_strain_stress(sim_strain, sim_stress, exp_strain, exp_stress, 
					   sim_YSx=None, sim_YSy=None, exp_YSx=None, exp_YSy=None,
					   smooth_exp=True, save_path=None, ax=None, **fig_kwargs):
	"""
	Plot homogenized stress strain curve.

	Args:
		sim_strain: The simulated strain.
		sim_stress: The simulated stress.
		exp_strain: The experimental strain.
		exp_stress: The experimental stress.
		sim_YSx: The simulated yield stress x.
		sim_YSy: The simulated yield stress y.
		exp_YSx: The experimental yield stress x.
		exp_YSy: The experimental yield stress y.
		smooth_exp: Whether to smooth the experimental data.
		save_path: The path to save the figure.
		ax: The axis to plot the figure on. If not specified, a new figure will be created.
		**fig_kwargs: Keyword arguments for the figure.
	"""
	ms = fig_kwargs.get('markersize', 5)
	me = fig_kwargs.get('markevery', 50)
	xlb = fig_kwargs.get('xlabel', 'plastic strain $\\varepsilon-\\varepsilon_y$')
	ylb = fig_kwargs.get('ylabel', 'stress $\sigma-\sigma_y$[MPa]')
	ylm = fig_kwargs.get('ylim', [0, 80])

	if smooth_exp:
		exp_strain = smoothing(exp_strain, 10)
		exp_stress = smoothing(exp_stress, 10)

	if not ax:
		fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)

	ax.plot(exp_strain, exp_stress, '--ok', alpha=0.8, label='exp', markersize=ms, markevery=me)
	ax.plot(sim_strain, sim_stress, c='red', alpha=0.8, label='approx.')

	if exp_YSx and exp_YSy:
		ax.scatter(exp_YSx, exp_YSy, c='k', s=10, label=f'{exp_YSy:.2f}MPa, exp')
	if sim_YSx and sim_YSy:
		ax.scatter(sim_YSx, sim_YSy, c='red', s=10, label=f'{sim_YSy:.2f}MPa, approx.') 

	if not ax:
		ax.grid(alpha=0.2)
		ax.set_xlabel(xlb)
		ax.set_ylabel(ylb)
		ax.set_ylim(ylm)
		ax.legend(loc='upper left')

	if save_path:
		plt.savefig(save_path)

def plot_dislocation_density(sim_strain, rho_ssd, rho_gnd, save_path=None):
	"""
	Plot the dislocation density evolution.

	Args:
		sim_strain: The simulated plastic strain.
		rho_ssd: The simulated SSD dislocation density.
		rho_gnd: The simulated GND dislocation density.
		save_path: The path to save the figure.
	"""
	_, ax = plt.subplots(figsize=(6,4), tight_layout=True)
	ax.grid(alpha=0.2)
	ax.plot(sim_strain, rho_ssd, c='b', alpha=0.8, linestyle='dashed', label='$\\rho_{ssd}$')
	ax.plot(sim_strain, rho_gnd, c='r', alpha=0.8, linestyle='dashed', label='$\\rho_{gnd}$')
	ax.plot(sim_strain, rho_ssd+rho_gnd, c='k', alpha=0.8, label='$\\rho$')
	ax.set_xlabel('plastic strain $\\varepsilon-\\varepsilon_y$')
	ax.set_ylabel('dislocation density $\\rho$[$m^{-2}$]')
	ax.set_yscale('log')
	ax.set_ylim(1e10,1e15)
	ax.legend(loc='lower right')
	if save_path:
		plt.savefig(save_path)

# >----------------------------------------------------------------------------------------------------
# > legacy code, need update
# >----------------------------------------------------------------------------------------------------
def plotDAMASK3_eval(handler, broken=True):
	"""
	Plot the predicted vs. measured stress for DAMASK3.

	Warning:
		legacy code

	Args:
		handler: The DAMASK3 handler object.
		broken: Whether to remove the initial yield offset caused by exponent n.
	"""
	fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
	markers = ['o','^','s','D']
	labels = ['7min','30min','6hr','168hr']
	colors = iter(['cyan', 'yellow', 'orange', 'red'])

	for i, _ in enumerate(handler.params_updater()):

		# wh
		exp_strain, exp_stress, exp_YSx, exp_YSy = get_plastic(*handler.getEXP(), flow_stress=False)
		sim_strain, sim_stress, sim_YSx, sim_YSy = get_plastic(*handler.getDAMASK3(), flow_stress=False)
		measure_stress = get_uniform_stress(exp_strain, exp_stress)
		calc_stress = get_uniform_stress(sim_strain, sim_stress)

		# TODO: align calc_stress with calc_YS, due to the exponent n >= 1
		if broken:
			handler.c_params()
			handler.g_params()
			calc_YS = handler.prm.YS/1e6
			calc_stress -= calc_stress[0] - calc_YS

		# change color from blue -> red, with temperature increasing 
		if handler.prm.cond=='7min':
			ax.scatter(calc_stress, measure_stress, c=next(colors), marker='o', s=25, edgecolors='k', linewidth=.8)
		else:
			ax.scatter(calc_stress, measure_stress, c='cyan', marker=markers[i], s=25, edgecolors='k', linewidth=.8)

	ylim = 400
	ylim_bound = ylim*0.1	# 10% error 
	ax.plot([0,ylim], [0,ylim], c='k', label='1:1', alpha=0.6)
	ax.plot([0,ylim], [0,ylim-ylim_bound], 'k--', alpha=0.6)
	ax.plot([0,ylim-ylim_bound], [0,ylim], 'k--', alpha=0.6)

	ax.set_xlabel('calculated stress [MPa]')
	ax.set_ylabel('measured stress [MPa]')
	ax.set_xlim(0,ylim)
	ax.set_ylim(0,ylim)
	ax.grid(alpha=0.2)

	# legend 
	dummys = [
		ax.scatter([], [], marker=marker, color='none', label=label, s=25, edgecolors='k', linewidth=.8)
		for marker, label in zip(markers, labels)
	]
	ax.legend(handles=dummys, loc='lower right')
	plt.savefig(f'{handler.prm.WORKdir}/{handler.prm.TASKname}_eval.png', dpi=300)


def plotDAMASK3_allInOne(handler, plastic_strain=False):
	"""
	plot homogenized stress strain curve over all of the aging conditions

	Warning:
		legacy code
	"""
	plt.rc('font', size=25)
	_, ax1 = plt.subplots(figsize=(16,10))
	plt.grid()
	colors = ['cyan', 'salmon', 'limegreen', 'violet', 'orange' ,'navy', 'darkslategray']
	for c, _ in zip(colors, handler.params_updater()):
		exp_strain, exp_stress, exp_YSx, exp_YSy = handler.getEXP()
		sim_strain, sim_stress, sim_YSx, sim_YSy = handler.getDAMASK3()
		if plastic_strain:
			# shift to plastic strain
			sim_stress = sim_stress[sim_strain>=sim_YSx]
			sim_stress -= sim_stress[0] - exp_YSy
			sim_strain = sim_strain[sim_strain>=sim_YSx]
			sim_strain -= sim_strain[0]

			exp_stress = exp_stress[exp_strain>=exp_YSx]
			exp_strain = exp_strain[exp_strain>=exp_YSx]
			exp_strain -= exp_strain[0]
			if exp_YSy[0]: ax1.scatter(0, exp_YSy[0], c=f'{c}', label=f'{exp_YSy[0]:.2f}MPa, EXP')
		else:
			if exp_YSy[0]: ax1.scatter(exp_YSx[0], exp_YSy[0], c=f'{c}', label=f'{exp_YSy[0]:.2f}MPa, EXP')
			if sim_YSy[0]: ax1.scatter(sim_YSx[0], sim_YSy[0], c=f'{c}', label=f'{sim_YSy[0]:.2f}MPa, {handler.prm.cond}_{handler.prm.T}K') 

		ax1.plot(exp_strain, exp_stress, c=f'{c}', linestyle='--')
		ax1.plot(sim_strain, sim_stress, c=f'{c}', linewidth=3)

	xlabel = 'plastic strain (_)' if plastic_strain else 'strain (_)'
	ylabel = 'flow stress (MPa)' if plastic_strain else 'stress (MPa)'
	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel)
	ax1.set_ylim(0,350)
	ax1.set_xlim(0,0.10)
	title = handler.prm.TASKname.split('_')[-1]
	ax1.set_title(f'{title}')
	ax1.legend(loc='lower right')
	plt.savefig(f'{handler.prm.WORKdir}/{handler.prm.TASKname}_allInOne.png')

def plotKinetics(handler):
	"""
	Warning:
		legacy code
	"""
	Ts = np.arange(298, 698, 10)
	# stable
	stable_YSs = []
	handler.prm.c_PK = []
	for T in Ts:
		handler.prm.T = T
		handler.c_params()
		stable_YSs.append(handler.prm.YS)
		
	# nonstable
	nonstable_YSs = []
	handler.prm.c_PK = ['c_kinetics']
	for T in Ts:
		handler.prm.T = T
		handler.c_params()
		nonstable_YSs.append(handler.prm.YS)
		
	stable_YSs = np.array(stable_YSs)/1e6
	nonstable_YSs = np.array(nonstable_YSs)/1e6

	_, ax = plt.subplots(figsize=(6,4), tight_layout=True)
	ax.plot(Ts, stable_YSs, '--', color='blue', label='stable')
	ax.plot(Ts, nonstable_YSs, color='red', label='non-stable')
	ax.scatter([298, 423, 473, 523], [290.59, 243.79, 224.36, 155.73], c='k', s=15, label='exp')
	ax.set_xlabel('temperature[K]')
	ax.set_ylabel('yield stress[MPa]')
	ax.legend()
	ax.grid(alpha=0.2)
	plt.savefig(f'{handler.prm.WORKdir}/{handler.prm.TASKname}_kinetics.png', dpi=300)

def plotDAMASK3_hardrate(handler):
	"""
	Warning:
		legacy code
	"""
	exp_strain, exp_stress, exp_YSx, exp_YSy = handler.getEXP(plastic=True)
	sim_strain, sim_stress, sim_YSx, sim_YSy = handler.getDAMASK3(plastic=True)

	window_width = 100
	exp_strain = smoothing(exp_strain, window_width)
	exp_stress = smoothing(exp_stress, window_width)
	exp_hardrate = np.diff(exp_stress) / np.diff(exp_strain)
	eng_hardrate = np.diff(sim_stress) / np.diff(sim_strain)

	plt.plot(exp_strain[1:], exp_hardrate, 'k--')
	plt.plot(sim_strain[1:], eng_hardrate, c='salmon')
	plt.ylim(0,np.max(exp_hardrate))
	plt.xlabel('strain')
	plt.ylabel('work hardening rate')
	plt.title(f'{handler.prm.cond}_{handler.prm.T}')
	plt.savefig(f'{handler.prm.WORKdir}/{handler.prm.TASKname}_hardrate.png')
	plt.close()