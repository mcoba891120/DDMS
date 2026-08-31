import os
import numpy as np
import pandas as pd
import matplotlib
try:
	matplotlib.use('tkAgg')
except ImportError:
	matplotlib.use('Agg')	# no GUI backend available (e.g. headless/CI)
import matplotlib.pyplot as plt
from tqdm import tqdm

# >----------------------------------------------------------------------------------------------------
# > TEM analysis
# >----------------------------------------------------------------------------------------------------
def c_TEM_6111(prm):
	EXPname 	= prm.EXPname
	thickness 	= prm.thickness*1e9  	# (nm)
	classes		= prm.classes			# (_)
	r_min		= prm.r_min*1e9			# (nm)
	r_max		= prm.r_max*1e9			# (nm)
	rtrans 		= prm.rtrans*1e9        # (nm)
	rcl			= prm.rcl*1e9			# (nm)
	rl 			= prm.rl*1e9 			# (nm)
	ManualAutoFactor = 0.78 			# manually/automatically, factor to fit auto threshold method
	VolumeFactor = 1.2					# TEM volume factor

	# TY format
	data = pd.read_csv(EXPname)
	for col in data.columns:
		if data[col].dropna().values.size<1:
			data = data.drop(col, axis=1)

	# needle-like 
	tem_areas = np.array([float(e.split('_')[-1]) for e in data.columns]) 	# area of TEM images (nm^2)
	v = (thickness+rl)*tem_areas*VolumeFactor								# volume of virtual box containing precipitates (nm^3)
	a = data.to_numpy()*ManualAutoFactor
	a = np.ma.array(a, mask=np.isnan(a))
	r = np.sqrt(a/np.pi)
	n = np.array([a_col[~np.isnan(a_col)].size for a_col in a.T])

	# spherical equivalent
	rl_dist	= rl*r/r.mean()													# artificial rl, same distribution as r
	req = np.power(np.power(r, 2)*rl_dist*3/4, 1/3)
	roeq = np.where(req>=rtrans, req, float("nan"))
	roeq = np.ma.array(roeq, mask=np.isnan(roeq))
	no = np.array([r_col[~np.isnan(r_col)].size for r_col in roeq.T])

	# discretize
	r_i = np.linspace(r_min, r_max, classes) 								# radius of each class
	Nv_i = np.zeros_like(r_i) 												# number density of each class
	for i in range(len(r_i)):
		upper_bool = req<r_i[i+1] if i!=len(r_i)-1 else req<-np.inf
		lower_bool = req>=r_i[i]
		Ni = 3*np.count_nonzero(np.logical_and(lower_bool, upper_bool))
		Nv_i[i] = Ni/np.sum(v)
	Nvo_i = np.where(r_i>=rtrans, Nv_i, np.zeros_like(r_i))					# number density of each class for Orowan particles

	f_all = np.sum(4/3*np.pi*np.power(r_i, 3)*Nv_i)
	f_o = np.sum(4/3*np.pi*np.power(r_i, 3)*Nvo_i)
	phi = np.clip((req-rtrans)/(rcl-rtrans), 0, 1).mean()
	lmb = 1/(8*np.sum(np.power(r_i[r_i>=rtrans], 2)*Nv_i[r_i>=rtrans]))
	lmb_star = lmb/phi if phi>0 else np.inf

	# >--------------------------------------------------
	# legacy
	# Nv = 3*np.sum(n)/np.sum(v)
	# Nvo = 3*np.sum(no)/np.sum(v)
	# f_all = Nv*np.mean(a)*rl
	# f_all = Nv*4/3*np.pi*np.power(req.mean(), 3)
	# f_o = Nvo*4/3*np.pi*np.power(roeq.mean(), 3)

	# # bootstrap	
	# bootstrap = 100	
	# A = np.mean(a, axis=0)
	# A_bts = np.mean(np.random.choice(A, size=(bootstrap, len(A))), axis=1)
	# A_spread = np.std(A_bts)
	# >--------------------------------------------------

	prm.R_mean		= r.mean()*1e-9
	prm.Req_mean	= req.mean()*1e-9											# equivalent precipitate mean radius (nm->m)
	prm.R_std		= np.std(np.mean(req, axis=0))*1e-9
	prm.f_all 		= f_all	 													# volume fraction of all precipitates (_)
	prm.f_o 		= f_o														# volume fraction of Orowan precipitates, from Larry (_)
	prm.No_mean 	= np.mean(no) 												# average number of Orowan precipitates (1 direction)
	prm.lmb 		= lmb*1e-9 													# average plasticity slip distance, for GND calculation (nm->m)
	prm.lmb_star 	= lmb_star*1e-9 											# effective plasticity slip distance, for GND calculation (nm->m)

	prm.tem_areas	= tem_areas*1e-18  											# (nm^2->m^2)
	prm.V			= np.sum(v)*1e-27											# total TEM volume (nm^3->m^3)
	prm.r			= req.ravel()*1e-9											# equivalent particle size distribution (nm->m)
	prm.r_i			= r_i*1e-9													# discretized particle radius (nm->m)
	prm.Nv_i		= Nv_i*1e27													# discretized number density (nm^-3->m^-3)
	prm.Nvo_i		= Nvo_i*1e27												# (nm^-3->m^-3)
	prm.Nv 			= Nv_i.sum()*1e27  											# (nm^-3->m^-3)
	prm.Nvo			= Nvo_i.sum()*1e27											# (nm^-3->m^-3)
	prm.Nv_std 		= np.std(3*n/v)*1e27
	prm.phi 		= phi														# Orowan efficiency factor
	return prm

def c_TEM_7075(prm):
	"""
	Myriam Nicolas. Precipitation evolution in an Al-Zn-Mg 
	alloy during non-isothermal heat treatments and in the 
	heat-affected zone of welded joints. Material chemistry.2002.
	"""
	EXPnames 	= prm.EXPnames
	thickness 	= prm.thickness*1e9 # (nm)

	###################### TO UPDATE ######################
	edge = 123											# length of crops of TEM images (nm)
	V = thickness*edge**2 								# investigated volume (nm^3)
	EXPnames.sort(key=lambda x: x.split('_')[0])
	###################### TO UPDATE ######################

	R_sphere = np.array([]); R_platlet = np.array([]);
	v_sphere = np.array([]); v_platlet = np.array([]);
	v_correction = np.array([]);
	N_sphere = np.array([]); N_platlet = np.array([]);
	for EXPname in EXPnames:
		data = pd.read_csv(EXPname)
		rmax = data['Feret'].values/2
		rmin = data['MinFeret'].values/2
		n = len(rmax)
		if 'sphere' in EXPname:
			rmean = np.mean(0.5*(rmax+rmin))
			vmean = (4/3)*np.pi*rmean**3*n
			R_sphere = np.append(R_sphere, rmean)
			v_sphere = np.append(v_sphere, vmean)
			N_sphere = np.append(N_sphere, n)
		else:
			rmean = np.mean(rmax)
			vmean = np.mean(rmin)*np.pi*np.mean(rmax)**2*n
			R_platlet = np.append(R_platlet, rmean)
			v_platlet = np.append(v_platlet, vmean)
			N_platlet = np.append(N_platlet, n)
			v_correction = np.append(v_correction, (4/3)*np.pi*rmean**3*n)

	R = np.mean([R_sphere, R_platlet], axis=0)
	vs = v_sphere+2*v_platlet-np.mean(v_correction)
	fs = vs/V
	N = No = N_sphere+N_platlet
	rbar = np.sqrt(2/3)*R
	lamda = 2*rbar*(np.sqrt(np.pi/(4*fs))-1)

	# bootstrap	
	bootstrap = 100	
	c = [np.random.choice(len(R), len(R)) for _ in range(bootstrap)]
	R_bts = [np.mean(R[cc]) for cc in c]
	f_bts = [np.mean(fs[cc]) for cc in c]
	No_bts = [np.mean(No[cc]) for cc in c]
	lamda_bts = [np.mean(lamda[cc]) for cc in c]
	rbar_bts = [np.mean(rbar[cc]) for cc in c]

	prm.R_mean 	= np.mean(R_bts)*1e-9		# mean radius of all precipitate (nm->m)
	prm.f_all 	= np.mean(f_bts)	 		# volume fraction of all precipitates (_)
	prm.f_std 	= 2*np.std(f_bts) 			# standard deviation of f_all

	prm.f_o 	= prm.f_all					# volume fraction of Orowan precipitates, from Larry
	prm.fo_std 	= 2*prm.f_std 				# standard deviation of f_o
	prm.No_mean = np.mean(No_bts) 			# average number of Orowan precipitates (1 direction)
	prm.rbar  	= np.mean(rbar_bts)
	prm.lamda 	= np.mean(lamda_bts)*1e-9	# average plasticity slip distance, for GND calculation (nm->m)
	# prm.r 	= r.ravel()*1e-9			# particle size distribution (nm->m)
	# prm.V		= np.sum(v)*1e-27			# total TEM volume (nm^3->m^3)
	prm.Nv_sum 	= N/V*1e27  				# (nm^-3->m^-3)
	prm.tem_areas= np.array([edge**2*1e-18 for _ in range(len(EXPnames)//2)])  # (nm^2->m^2)
	return prm

# >----------------------------------------------------------------------------------------------------
# > precipitation kinetics
# >----------------------------------------------------------------------------------------------------
def c_kinetics(prm, debug=False, W=False):
	"""
	MODELLING OF NON-ISOTHERMAL TRANSFORMATIONS
	IN ALLOYS CONTAINING A PARTICLE DISTRIBUTION
	"""
	classes	= prm.classes
	r_min 	= prm.r_min  			# (m)
	r_max 	= prm.r_max				# (m)
	r_i		= prm.r_i				# (m)
	Nv_i	= prm.Nv_i				# (m^-3)
	rtrans 	= prm.rtrans			# (m)
	rcl 	= prm.rcl				# (m)
	rl 		= prm.rl 				# (m)
	wt 		= np.array(prm.wt) 		# (%)
	aw 		= np.array(prm.aw) 		#
	cc 		= prm.cc
	RR 		= prm.RR
	rate 	= prm.rate
	V 		= prm.V    	 			# (m^3)
	T 		= prm.T  				#
	Cp 		= prm.Cp				# (wt%)
	A0 		= prm.A0				# 
	j0 		= prm.j0				# 
	D0 		= prm.D0				# (m^2/s)
	Qd 		= prm.Qd				# 
	Cs 		= prm.Cs
	Qs 		= prm.Qs
	sigma 	= prm.sigma		
	Vm 		= prm.Vm				# (m^3/mol)

	if prm.cond=='10min':
		W = True
		T = 273 + 180

	########## initialize ##########
	delta_r	= (r_max-r_min)/classes 					# length of control volume
	D 		= D0*np.exp(-Qd/(RR*T)) 					# diffusion coefficient
	C0 		= wt[2]  									# initial Mg solute concentration in matrix (wt%)
	Cbar 	= cc[0]										# start from 7min, current Mg solute concentration in matrix (wt%)
	Ce 		= Cs*np.exp(-Qs/(RR*T))						# equilibrium solute content at particle/matrix interface)
	Ci 		= Ce*np.exp(2*sigma*Vm/(r_i*RR*T)) 			# interface velocity, spherical
	# A 		= 0.5*rl/np.mean(r) 					# aspect ratio
	# Ci = Ce*(1+(1+1/A)*sigma*Vm/(RR*T*ri)*(1-Ce)/Cp)	# interface velocity, needle-like
	if W:												# if start from W condition
		Cbar = C0
		Nv_i = np.zeros_like(r_i)

	########## evolve ##########
	strain = prm.YS/69e9+0.002 	# assuming yield
	# strain = 7*60*rate			# 7min
	# strain = 60*60*rate				# 10min
	# strain = 30*60*rate 			# 30min
	# strain = 6*60*60*rate 		# 6hr
	# strain = 168*60*60*rate 		# 168hr
	numerical_inc = 1e-10 		# to prevent inf/nan
	delta_t = 1e-1
	times = np.arange(0, strain/rate+delta_t, delta_t)

	if debug:
		Nv_history = np.empty_like(times)
		R_history = np.empty_like(times)
		j_history = np.empty_like(times)
		rstar_history = np.empty_like(times)
		Cbar_history = np.empty_like(times)
		f_history = np.empty_like(times)

		# plt.figure()
		# plt.xlabel('radius[nm]')
		# plt.ylabel('normalized distribution[$nm^{-1}$]')
		# plt.ylabel('normalized distribution[$nm^{-1}$]')
		# plt.xlim(0, 15)
		# plt.ylim(0, 0.25)
		# plt.xticks(np.linspace(0, 16, 9))
		# plt.yticks(np.linspace(0, 0.25, 6))

	for i, t in enumerate(tqdm(times, desc='<<< kinetics >>>')):
		if debug:
			if t==0:
				# plt.hist(r_i[Nv_i>0]/1e-9, weights=Nv_i[Nv_i>0], bins=10, alpha=0.6)
				# plt.hist(prm.r/1e-9, weights=prm.r/prm.r.sum(), bins=10, alpha=0.6, color='orange')
				# plt.hist(r_i[Nv_i>0]/1e-9, weights=Nv_i[Nv_i>0], bins=20, alpha=0.6, density=True)
				pass

			if np.abs(t-7*60) < delta_t/2:
				# plt.figure()
				# plt.hist(r_i[Nv_>0]/1e-9, weights=Nv_[Nv_>0], bins=10, alpha=0.6, density=True)

				# x = r_i/1e-9
				# y = Nv_i/(Nv_i*(delta_r/1e-9)).sum()
				# # mask = np.logical_and(x>(r_i[Nv_>0]/1e-9).min(), x<(r_i[Nv_>0]/1e-9).max())
				# mask = x<(r_i[Nv_>0]/1e-9).max()
				# plt.plot(x[mask], y[mask], color='red', alpha=0.8)
				# plt.legend(['fit', 'exp'], loc='upper right', title='7min at $250^{\circ} C$')
				pass

			if np.abs(t-30*60) < delta_t/2:
				# plt.figure()
				# plt.hist(r_i[Nv_>0]/1e-9, weights=Nv_[Nv_>0], bins=10, alpha=0.6, density=True)

				# x = r_i/1e-9
				# y = Nv_i/(Nv_i*(delta_r/1e-9)).sum()
				# # mask = np.logical_and(x>(r_i[Nv_>0]/1e-9).min(), x<(r_i[Nv_>0]/1e-9).max())
				# mask = x<(r_i[Nv_>0]/1e-9).max()
				# plt.plot(x[mask], y[mask], color='red', alpha=0.8)
				# plt.legend(['fit', 'exp'], loc='upper right', title='30min at $250^{\circ} C$')
				pass

			if np.abs(t-6*60*60) < delta_t/2:
				# plt.figure()
				# plt.hist(r_i[Nv_>0]/1e-9, weights=Nv_[Nv_>0], bins=10, alpha=0.6, density=True)

				# x = r_i/1e-9
				# y = Nv_i/(Nv_i*(delta_r/1e-9)).sum()
				# # mask = np.logical_and(x>(r_i[Nv_>0]/1e-9).min(), x<(r_i[Nv_>0]/1e-9).max())
				# mask = x<(r_i[Nv_>0]/1e-9).max()
				# plt.plot(x[mask], y[mask], color='red', alpha=0.8)
				# plt.legend(['fit', 'exp'], loc='upper right', title='6hr at $250^{\circ} C$')
				pass

			if np.abs(t-168*60*60) < delta_t/2:
				# plt.figure()
				# plt.hist(r_i[Nv_>0]/1e-9, weights=Nv_[Nv_>0], bins=10, alpha=0.6, density=True)

				# x = r_i/1e-9
				# y = Nv_i/(Nv_i*(delta_r/1e-9)).sum()
				# # mask = np.logical_and(x>(r_i[Nv_>0]/1e-9).min(), x<(r_i[Nv_>0]/1e-9).max())
				# mask = x<(r_i[Nv_>0]/1e-9).max()
				# plt.plot(x[mask], y[mask], color='red', alpha=0.8)
				# plt.legend(['fit', 'exp'], loc='upper right', title='168hr at $250^{\circ} C$')
				pass

			if t==times[-1]:
				# plt.legend(loc='upper right')
				# plt.legend(['exp'], loc='upper right', title='7min at $250^{\circ} C$')
				# plt.xlabel('radius[nm]')
				# plt.ylabel('normalized distribution[$nm^{-1}$]')
				# plt.grid(alpha=0.2)
				# plt.show()
				pass

		########## nucleation rate ##########
		G_het = A0**3/(RR*T*np.log(Cbar/Ce))**2 													# spherical
		# G_het = 8*np.pi/27*(2*A+1)**3/A**2*Vm**2*sigma**3/(RR*T*np.log(Cbar/Ce))**2  				# needle-like
		j = j0*np.exp(-G_het/(RR*T))*np.exp(-Qd/(RR*T))
		
		########## add particles at rstar + delta_rstar ##########
		rstar = 2*sigma*Vm/(RR*T)/np.log(Cbar/Ce) 													# spherical
		# rstar = 2/3*Vm*sigma/(RR*T)*(2*A+1)/A/np.log(Cbar/Ce) 									# needle-like
		index = np.where(r_i<=(rstar+0.05*rstar))[0]
		if len(index)!=0:
			Nv_i[index[-1]] += j*delta_t

		########## mass balance ##########
		Req_mean = np.sum(Nv_i*r_i)/np.sum(Nv_i)
		f_all = np.sum(4/3*np.pi*np.power(r_i, 3)*Nv_i) 											# spherical
		# f_all = np.sum(np.pi*np.power(r_i, 2)*(2*A*R_mean)*Nv) 									# needle-like
		# f_all = np.minimum(f_all, (C0-Ce-numerical_inc)/(Cp-Cbar))
		# Cbar = np.maximum(Ce+numerical_inc, C0-(Cp-Cbar)*f_all)
		Cbar = np.maximum(Ce+numerical_inc, (C0-Cp*f_all)/(1-f_all))

		########## growth rate ##########
		v = (Cbar-Ci)/(Cp-Ci)*D/r_i 																# spherical
		# v = np.where(Cbar>Ci, \
		# 		1/3*np.sqrt((Cbar-Ci)/(Cp-Ci)*D/np.pi/A/(t+numerical_inc)), \
				# 2/9*D/np.pi*sigma*Vm/(RR*T)*(1+A)/A**2*Ce*(1-Ce)/Cp**2*(ri-R_mean)/ri**2/R_mean) 	# needle-like

		# if t%500==0:
		# 	print(v[:20])
		# 	print(Ci[:20])
		# 	print(Cbar)
		########## update coefficients ##########
		ve = np.append(v[1:], numerical_inc)
		vw = v

		ap0 = delta_r/delta_t
		ae = np.zeros_like(Nv_i)
		aw = np.zeros_like(Nv_i)
		ap = np.zeros_like(Nv_i)

		vevw_pospos = np.logical_and(ve>0, vw>0)
		vevw_posneg = np.logical_and(ve>0, vw<0)
		vevw_negpos = np.logical_and(ve<0, vw>0)
		vevw_negneg = np.logical_and(ve<0, vw<0)

		aw[vevw_pospos] = vw[vevw_pospos]
		ap[vevw_pospos] = ap0 + ve[vevw_pospos]
		ap[vevw_posneg] = ap0 + ve[vevw_posneg] - vw[vevw_posneg]
		ae[vevw_negpos] = -ve[vevw_negpos]
		aw[vevw_negpos] = vw[vevw_negpos]
		ap[vevw_negpos] = ap0
		ae[vevw_negneg] = -ve[vevw_negneg]
		ap[vevw_negneg] = ap0 - vw[vevw_negneg]

		Nv_i = (ae*np.append(Nv_i[1:],0) + aw*np.append(0,Nv_i[:-1]) + ap0*Nv_i)/ap

		if debug:
			Nv_history[i] = Nv_i.sum()
			j_history[i] = j
			R_history[i] = Req_mean
			rstar_history[i] = rstar
			Cbar_history[i] = Cbar
			f_history[i] = f_all
	
	if debug:
		pass
		# >--------------------------------------------------
		# experiment data
		ts = np.array([7*60, 30*60, 360*60, 168*60*60])				# (s)
		if not W:
			ts -= 7*60
		# rs = np.array([1.77, 2.60, 3.77, 7.56])					# (nm)
		rs = np.array([5.99, 8.84, 14.48, 24.86])					# equivalent r (nm)
		rs_std = np.array([0.30, 0.61, 1.25, 3.61])					# (nm)
		nvs = np.array([151.22, 45.57, 9.88, 1.76])*1e20			# (m^-3)
		nvs_std = np.array([26.98, 12.49, 2.3, 0.87])*1e20			# (m^-3)
		fs = np.array([1.52, 1.70, 1.51, 1.44])/100					# (_)

		# fig, ax1 = plt.subplots(figsize=(6, 4.5), tight_layout=True)
		# ax1.plot(times, Nv_history, c='b', label='number density')
		# ax1.errorbar(ts, nvs, yerr=nvs_std, c='b', fmt='o', capsize=3)
		# ax1.set_xlim(1e0, 1e6)
		# ax1.set_ylim(1e18,1e24)
		# ax1.set_yticks(np.linspace(1e18, 1e24, 7))
		# ax1.set_xscale('log')
		# ax1.set_yscale('log')
		# ax1.set_xlabel('time[s]')
		# ax1.set_ylabel('number density[$m^{-3}$]')
		# ax1.grid(alpha=0.2)
		
		# ax2 = ax1.twinx()
		# ax2.plot(times, R_history*1e9, c='r', label='mean radius')
		# ax2.errorbar(ts, rs, yerr=rs_std, c='r', fmt='o', capsize=3)
		# ax2.set_ylim(0, 30)
		# ax2.set_yticks(np.linspace(0, 30, 7))
		# ax2.set_ylabel('mean radius[nm]')
		# fig.legend(bbox_to_anchor=(0.90, 0.28))
		# # plt.show()
		# # >--------------------------------------------------
		# plt.figure()
		# plt.plot(times, Cbar_history, label='Cbar')
		# plt.plot(times, np.ones_like(times)*Ce, label='Ce')
		# plt.plot(times, np.ones_like(times)*C0, label='C0')
		# plt.plot(times, np.ones_like(times)*Ci.mean(), label='Ci')
		# plt.legend()
		# plt.ylim(0, 1)
		# plt.ylabel('concentration')
		# plt.xscale('log')

		# plt.figure()
		# plt.plot(times, f_history)
		# plt.ylim(0, 0.1)
		# plt.ylabel('volume fraction')
		# plt.show()
		# >--------------------------------------------------
		fig, axes = plt.subplots(3, 1, figsize=(6, 4.5*3), tight_layout=True, gridspec_kw = {'hspace':0})
		# optional overlay of a previously dumped run for comparison;
		# set DDMS_KINETICS_OVERLAY_DIR to a directory containing
		# Nv_history/j_history/R_history/rstar_history/f_history .npy dumps to enable
		overlay_dir = os.environ.get('DDMS_KINETICS_OVERLAY_DIR', '')
		if overlay_dir:
			_Nv_history = np.load(f"{overlay_dir}/Nv_history", allow_pickle=True)
			_j_history = np.load(f"{overlay_dir}/j_history", allow_pickle=True)
			_R_history = np.load(f"{overlay_dir}/R_history", allow_pickle=True)
			_rstar_history = np.load(f"{overlay_dir}/rstar_history", allow_pickle=True)
			_f_history = np.load(f"{overlay_dir}/f_history", allow_pickle=True)
			_times = np.linspace(0, 168*60*60, len(_Nv_history))
		else:
			_Nv_history = _j_history = _R_history = _rstar_history = _f_history = np.array([])
			_times = np.array([])

		# >----------------------------------------
		# number density
		l1 = axes[0].plot(times, Nv_history, c='navy', label='number density')
		axes[0].plot(_times, _Nv_history, c='navy', alpha=0.2)
		axes[0].errorbar(ts, nvs, yerr=nvs_std, c='navy', fmt='o', capsize=3)
		axes[0].set_xscale('log')
		axes[0].set_yscale('log')
		axes[0].set_xlim(1e0, 1e6)
		axes[0].set_ylim(1e18,1e24)
		axes[0].set_xticklabels([])
		axes[0].set_yticks(np.array([10**(18+n) for n in range(7)], dtype=float))
		axes[0].set_ylabel('number density[$m^{-3}$]')
		axes[0].grid(alpha=0.2)

		ax_ = axes[0].twinx()
		l2 = ax_.plot(times, j_history, '--', c='navy', label='nucleation rate')
		ax_.plot(_times, _j_history, c='navy', alpha=0.2)
		ax_.set_ylim(1e18,1e24)
		ax_.set_yscale('log')
		ax_.set_yticks(np.array([10**(18+n) for n in range(7)], dtype=float))
		ax_.set_ylabel('nucleation rate[$m^{-3}s$]')
		lns = l1 + l2
		labs = [l.get_label() for l in lns]
		ax_.legend(lns, labs, loc='upper left')

		# >----------------------------------------
		# mean radius
		axes[1].plot(times, R_history*1e9, c='r', label='mean radius')
		axes[1].plot(times, rstar_history*1e9, '--', c='r', label='critical radius')
		axes[1].plot(_times, _R_history*1e9, c='r', alpha=0.2)
		axes[1].plot(_times, _rstar_history*1e9, '--', c='r', alpha=0.2)
		axes[1].errorbar(ts, rs, yerr=rs_std, c='r', fmt='o', capsize=3)
		axes[1].set_xscale('log')
		axes[1].set_xlim(1e0, 1e6)
		axes[1].set_ylim(0, 30)
		axes[1].set_xticklabels([])
		axes[1].set_yticks(np.linspace(0, 25, 6))
		axes[1].set_ylabel('mean radius[nm]')
		axes[1].grid(alpha=0.2)
		axes[1].legend(loc='upper left')
		
		# >----------------------------------------
		# volume fraction
		axes[2].plot(times, f_history*100, c='g', label='volume fraction')
		axes[2].plot(_times, _f_history*100, c='g', alpha=0.2)
		axes[2].scatter(ts, fs*100, c='g')
		axes[2].set_xlim(1e0, 1e6)
		axes[2].set_ylim(0, 2.4)
		axes[2].set_yticks(np.linspace(0, 2.0, 6))
		axes[2].set_xscale('log')
		axes[2].set_xlabel('time[s]')
		axes[2].set_ylabel('volume fraction[%]')
		axes[2].grid(alpha=0.2)

		for ax in [ax_, *axes]:
			ax.tick_params(which='major', axis='x', direction='in')
			ax.tick_params(which='major', axis='y', direction='in')
			ax.tick_params(which='minor', axis='x', direction='in')
			ax.tick_params(which='minor', axis='y', direction='in')

		for ax in axes:
			for t in ts:
				ax.plot([t, t], [0, 1e25], '--k', alpha=0.2)


		plt.show()
		# >--------------------------------------------------


	prm._Req_mean	= prm.Req_mean
	prm._Nv			= prm.Nv
	prm._Nvo		= prm.Nvo
	prm._f_all 		= prm.f_all
	prm._f_o 		= prm.f_o 
	prm._phi		= prm.phi
	prm._lmb 		= prm.lmb
	prm._lmb_star	= prm.lmb_star
	prm._ss 		= prm.ss 
	prm._pp 		= prm.pp 
	prm._YS 		= prm.YS

	prm.Req_mean 	= Req_mean  														# (m)
	prm.Nv_i 		= Nv_i																# (m^-3)
	prm.Nvo_i 		= np.where(r_i>=rtrans, Nv_i, np.zeros_like(r_i))					# (m^-3)
	prm.Nv 			= Nv_i.sum() 		 												# (m^-3)
	prm.Nvo 		= prm.Nvo_i.sum() 		 											# (m^-3)
	prm.f_all 		= f_all									 							# (_), 
	prm.f_o 		= np.sum(4/3*np.pi*np.power(r_i[r_i>=rtrans], 3)*Nv_i[r_i>=rtrans]) # (_), spherical
	prm.phi 		= (np.clip((r_i-rtrans)/(rcl-rtrans), 0, 1)*Nv_i).sum()/Nv_i.sum()	# (_)
	prm.lmb 		= 1/(8*np.sum(np.power(r_i[r_i>=rtrans], 2)*Nv_i[r_i>=rtrans]))		# (m)
	prm.lmb_star 	= prm.lmb/prm.phi if prm.phi>0 else np.inf							# (m)
	return prm

def c_kinetics_for_plot(prm, W=True):
	"""
	MODELLING OF NON-ISOTHERMAL TRANSFORMATIONS
	IN ALLOYS CONTAINING A PARTICLE DISTRIBUTION
	"""
	classes	= prm.classes
	r_min 	= prm.r_min  			# (m)
	r_max 	= prm.r_max				# (m)
	r_i		= prm.r_i				# (m)
	Nv_i	= prm.Nv_i				# (m^-3)
	rtrans 	= prm.rtrans			# (m)
	rcl 	= prm.rcl				# (m)
	rl 		= prm.rl 				# (m)
	wt 		= np.array(prm.wt) 		# (%)
	aw 		= np.array(prm.aw) 		#
	cc 		= prm.cc
	RR 		= prm.RR
	rate 	= prm.rate
	V 		= prm.V    	 			# (m^3)
	T 		= prm.T  				#
	Cp 		= prm.Cp				# (wt%)
	A0 		= prm.A0				# 
	j0 		= prm.j0				# 
	D0 		= prm.D0				# (m^2/s)
	Qd 		= prm.Qd				# 
	Cs 		= prm.Cs
	Qs 		= prm.Qs
	sigma 	= prm.sigma		
	Vm 		= prm.Vm				# (m^3/mol)

	########## initialize ##########
	delta_r	= (r_max-r_min)/classes 					# length of control volume
	D 		= D0*np.exp(-Qd/(RR*T)) 					# diffusion coefficient
	C0 		= wt[2]  									# initial Mg solute concentration in matrix (wt%)
	Cbar 	= cc[0]										# start from 7min, current Mg solute concentration in matrix (wt%)
	Ce 		= Cs*np.exp(-Qs/(RR*T))						# equilibrium solute content at particle/matrix interface)
	Ci 		= Ce*np.exp(2*sigma*Vm/(r_i*RR*T)) 			# interface velocity, spherical
	# A 		= 0.5*rl/np.mean(r) 					# aspect ratio
	# Ci = Ce*(1+(1+1/A)*sigma*Vm/(RR*T*ri)*(1-Ce)/Cp)	# interface velocity, needle-like
	if W:												# if start from W condition
		Cbar = C0
		Nv_i = np.zeros_like(r_i)

	########## evolve ##########
	# strain = prm.YS/69e9+0.002 	# assuming yield
	# strain = 7*60*rate			# 7min
	# strain = 30*60*rate 			# 30min
	# strain = 6*60*60*rate 		# 6hr
	strain = 168*60*60*rate 		# 168hr
	numerical_inc = 1e-10 		# to prevent inf/nan
	delta_t = 1e-1
	times = np.arange(0, strain/rate+delta_t, delta_t)

	Nv_history = np.empty_like(times)
	R_history = np.empty_like(times)
	j_history = np.empty_like(times)
	rstar_history = np.empty_like(times)
	Cbar_history = np.empty_like(times)
	f_history = np.empty_like(times)
	ss_history = np.empty_like(times)
	pp_history = np.empty_like(times)
	YS_history = np.empty_like(times)

	for i, t in enumerate(tqdm(times, desc='<<< kinetics >>>')):
		########## nucleation rate ##########
		G_het = A0**3/(RR*T*np.log(Cbar/Ce))**2 													# spherical
		# G_het = 8*np.pi/27*(2*A+1)**3/A**2*Vm**2*sigma**3/(RR*T*np.log(Cbar/Ce))**2  				# needle-like
		j = j0*np.exp(-G_het/(RR*T))*np.exp(-Qd/(RR*T))
		
		########## add particles at rstar + delta_rstar ##########
		rstar = 2*sigma*Vm/(RR*T)/np.log(Cbar/Ce) 													# spherical
		# rstar = 2/3*Vm*sigma/(RR*T)*(2*A+1)/A/np.log(Cbar/Ce) 									# needle-like
		index = np.where(r_i<=(rstar+0.05*rstar))[0]
		if len(index)!=0:
			Nv_i[index[-1]] += j*delta_t

		########## mass balance ##########
		Req_mean = np.sum(Nv_i*r_i)/np.sum(Nv_i)
		f_all = np.sum(4/3*np.pi*np.power(r_i, 3)*Nv_i) 											# spherical
		# f_all = np.sum(np.pi*np.power(r_i, 2)*(2*A*R_mean)*Nv) 									# needle-like
		# f_all = np.minimum(f_all, (C0-Ce-numerical_inc)/(Cp-Cbar))
		# Cbar = np.maximum(Ce+numerical_inc, C0-(Cp-Cbar)*f_all)
		Cbar = np.maximum(Ce+numerical_inc, (C0-Cp*f_all)/(1-f_all))

		########## growth rate ##########
		v = (Cbar-Ci)/(Cp-Ci)*D/r_i 																# spherical
		# v = np.where(Cbar>Ci, \
		# 		1/3*np.sqrt((Cbar-Ci)/(Cp-Ci)*D/np.pi/A/(t+numerical_inc)), \
				# 2/9*D/np.pi*sigma*Vm/(RR*T)*(1+A)/A**2*Ce*(1-Ce)/Cp**2*(ri-R_mean)/ri**2/R_mean) 	# needle-like

		########## update coefficients ##########
		ve = np.append(v[1:], numerical_inc)
		vw = v

		ap0 = delta_r/delta_t
		ae = np.zeros_like(Nv_i)
		aw = np.zeros_like(Nv_i)
		ap = np.zeros_like(Nv_i)

		vevw_pospos = np.logical_and(ve>0, vw>0)
		vevw_posneg = np.logical_and(ve>0, vw<0)
		vevw_negpos = np.logical_and(ve<0, vw>0)
		vevw_negneg = np.logical_and(ve<0, vw<0)

		aw[vevw_pospos] = vw[vevw_pospos]
		ap[vevw_pospos] = ap0 + ve[vevw_pospos]
		ap[vevw_posneg] = ap0 + ve[vevw_posneg] - vw[vevw_posneg]
		ae[vevw_negpos] = -ve[vevw_negpos]
		aw[vevw_negpos] = vw[vevw_negpos]
		ap[vevw_negpos] = ap0
		ae[vevw_negneg] = -ve[vevw_negneg]
		ap[vevw_negneg] = ap0 - vw[vevw_negneg]

		Nv_i = (ae*np.append(Nv_i[1:],0) + aw*np.append(0,Nv_i[:-1]) + ap0*Nv_i)/ap


		Nv_history[i] = Nv_i.sum()
		j_history[i] = j
		R_history[i] = Req_mean
		rstar_history[i] = rstar
		Cbar_history[i] = Cbar
		f_history[i] = f_all

		prm.Req_mean 	= Req_mean  														# (m)
		prm.Nv_i 		= Nv_i																# (m^-3)
		prm.Nvo_i 		= np.where(r_i>=rtrans, Nv_i, np.zeros_like(r_i))					# (m^-3)
		prm.Nv 			= Nv_i.sum() 		 												# (m^-3)
		prm.Nvo 		= prm.Nvo_i.sum() 		 											# (m^-3)
		prm.f_all 		= f_all									 							# (_), 
		prm.f_o 		= np.sum(4/3*np.pi*np.power(r_i[r_i>=rtrans], 3)*Nv_i[r_i>=rtrans]) # (_), spherical
		prm.phi 		= (np.clip((r_i-rtrans)/(rcl-rtrans), 0, 1)*Nv_i).sum()/Nv_i.sum()	# (_)
		prm.lmb 		= 1/(8*np.sum(np.power(r_i[r_i>=rtrans], 2)*Nv_i[r_i>=rtrans]))		# (m)
		prm.lmb_star 	= prm.lmb/prm.phi if prm.phi>0 else np.inf							# (m)

		########## update YS ##########
		prm = c_prec_6111_Myhr(prm)
		prm = c_ss_6111(prm)
		prm = c_YS_temper(prm, T=298)

		ss_history[i] = sum(prm.ss)
		pp_history[i] = prm.pp
		YS_history[i] = prm.YS

	print(ss_history)
	print(pp_history)
	print(YS_history)
	return prm


# >----------------------------------------------------------------------------------------------------
# > yield stress model
# >----------------------------------------------------------------------------------------------------
def c_G(prm):
	"""
	[1] Myhr, O.R., Hopperstad, O.S. & Børvik, T. A Combined 
	Precipitation, Yield Stress, and Work Hardening Model for 
	Al-Mg-Si Alloys Incorporating the Effects of Strain Rate 
	and Temperature. Metall Mater Trans A 49,(2018).
	"""
	T 		= prm.T 		# (K)
	u0 		= prm.u0 	 	# (Pa)
	Tm 		= prm.Tm 		# (K)
	theta 	= prm.theta 	# (_)

	G = u0*(1-T/Tm*np.exp(theta*(1-Tm/T)))

	prm.G = G 				# shear modulus (Pa)
	return prm

def c_prec_6111_Larry_avg(prm):
	"""
	[2] Larry et al. A new crystal plasticity constitutive model 
	for simulating precipitation-hardenable aluminum alloys, 2020.
	"""
	rtrans 	= prm.rtrans 		# (m)
	b 		= prm.b 		# (m)
	f_all 	= prm.f_all 	# (_)
	R_mean 	= prm.R_mean 	# (m)
	G 		= prm.G 		# (Pa)
	beta 	= prm.beta 		# (_)
	M 		= prm.M 		# (_)

	L = np.sqrt(2*np.pi/f_all)*R_mean	   						# center-to-center spacing of particles (nm)  

	Lf = np.where(R_mean<np.sqrt(3)*rtrans/2, \
				np.sqrt(np.sqrt(3)*rtrans/(2*R_mean))*L-2*R_mean,\
				L-2*R_mean) 									# effective mean spacing (nm)
	F = np.where(R_mean<rtrans, \
				2*beta*G*b**2*(R_mean/rtrans), \
				2*beta*G*b**2)
	pp = M*F/Lf/b 												# WARNING: should be M*F/Lf/b

	prm.pp = pp 												# sigma p, precipitate strengthening (Pa)

	# >--------------------------------------------------
	# plot 
	def get_tau_p(r, rtrans, mechanism='cut'):
		L = np.sqrt(2*np.pi/f_all)*r	   						# center-to-center spacing of particles (nm)  

		if mechanism=='cut':
			Lf = np.sqrt(np.sqrt(3)*rtrans/(2*r))*L-2*r
			F = 2*beta*G*b**2*(r/rtrans)
		if mechanism=='Orowan':
			Lf = L-2*r
			F = 2*beta*G*b**2
		return F/Lf/b
	
	ri = np.linspace(1e-9, 8e-9, 1000)
	tau_p_Oro = get_tau_p(ri, rtrans=0, mechanism='Orowan')
	tau_p_cut2 = get_tau_p(ri, rtrans=2e-9, mechanism='cut')
	tau_p_cut3 = get_tau_p(ri, rtrans=3e-9, mechanism='cut')
	tau_p_cut4 = get_tau_p(ri, rtrans=4e-9, mechanism='cut')

	fig, ax1 = plt.subplots(1, 1, figsize=(6,4), tight_layout=True)
	ax1.plot(ri/1e-9, tau_p_Oro/1e6, label='Orowan')
	ax1.plot(ri/1e-9, tau_p_cut2/1e6, label='cut, $r_c=2nm$')
	ax1.plot(ri/1e-9, tau_p_cut3/1e6, label='cut, $r_c=3nm$')
	ax1.plot(ri/1e-9, tau_p_cut4/1e6, label='cut, $r_c=4nm$')
	ax1.set_xlabel('radius[nm]')
	ax1.set_ylabel('shear stress[MPa]')
	ax1.set_yticks(np.linspace(0, 300, 6))
	ax1.set_ylim([0, 300])
	ax1.grid(alpha=0.2, axis='both')


	# discrete number density
	Nv = np.zeros_like(ri) 								# number density of each class
	for i in range(len(ri)):
		upper_bool = prm.r<ri[i+1] if i!=len(ri)-1 else prm.r<-999
		lower_bool = prm.r>=ri[i]
		Ni = 3*np.count_nonzero(np.logical_and(lower_bool, upper_bool)) # number of particles in class i
		Nv[i] = Ni/prm.V

	ax2 = ax1.twinx()
	# ax2.hist(prm.r/1e-9, weights=prm.r/prm.r.sum(), alpha=0.4, bins=10)
	ax2.hist(ri/1e-9, weights=Nv, bins=20, alpha=0.6, density=True)
	ax2.set_ylabel('normalized distribution[$nm^{-1}$]')
	# ax2.set_yticks(np.linspace(0, 0.3, 6))
	# ax2.set_ylim([0, 0.3])

	# plt.legend()
	plt.show()
	# >--------------------------------------------------

	return prm

def c_prec_6111_Larry(prm):
	"""
	[2] Larry et al. A new crystal plasticity constitutive model 
	for simulating precipitation-hardenable aluminum alloys, 2020.
	"""
	rtrans 	= prm.rtrans 		# (m)
	b 		= prm.b 		# (m)
	f_all 	= prm.f_all 	# (_)
	G 		= prm.G 		# (Pa)
	beta 	= prm.beta 		# (_)
	M 		= prm.M 		# (_)

	classes = 1000
	r_min 	= prm.r_min
	r_max 	= prm.r_max
	V 		= prm.V
	r 		= prm.r 		  		# (m)

	# discretization
	ri = np.linspace(r_min, r_max, classes) 			# radius of each class	
	Nv = np.zeros_like(ri) 								# number density of each class
	for i in range(len(ri)):
		upper_bool = r<ri[i+1] if i!=len(ri)-1 else r<-999
		lower_bool = r>=ri[i]
		Ni = 3*np.count_nonzero(np.logical_and(lower_bool, upper_bool)) # number of particles in class i
		Nv[i] = Ni/V

	c = np.sqrt(3)*rtrans/2
	Lf = np.where(
		ri<c,
		np.sqrt(c/ri) * np.sqrt(2*np.pi/f_all)*ri - 2*ri,
		np.sqrt(2*np.pi/f_all)*ri - 2*ri
	)
	F = np.where(ri<rtrans, 2*beta*G*b**2*(ri/rtrans), 2*beta*G*b**2)
	Lf_bar = (Nv*Lf).sum()/Nv.sum()
	F_bar = (Nv*F).sum()/Nv.sum()
	pp = M*F_bar/Lf_bar/b
	prm.pp = pp 												# sigma p, precipitate strengthening (Pa)
	return prm

def c_prec_6111_Myhr(prm):
	"""
	[3] O.R Myhr, Ø Grong, S.J Andersen, Modelling of the age 
	hardening behaviour of Al-Mg-Si alloys,	Acta Materialia, 2001.
	"""
	rtrans 		= prm.rtrans 	# (m)
	r_i 		= prm.r_i 		# (m)
	Nv_i		= prm.Nv_i		# (m^-3)
	f_all 		= prm.f_all 	# (_)
	Req_mean	= prm.Req_mean 	# (m)
	beta 		= prm.beta 		# (_)
	M 			= prm.M 		# (_)
	G 			= prm.G 		# (Pa)
	b 			= prm.b 		# (m)

	F = np.where(r_i<rtrans, 2*beta*G*b**2*(r_i/rtrans), 2*beta*G*b**2)
	F_bar = (Nv_i*F).sum()/Nv_i.sum()
	pp = M/(b*Req_mean)/np.sqrt(2*beta*G*b**2)* \
		np.sqrt(3*f_all/(2*np.pi))*np.power(F_bar, 3/2)

	prm.pp = pp 												# sigma p, precipitate strengthening (Pa)

	# >--------------------------------------------------
	# plot 
	# def get_tau_p(r, rtrans, mechanism='cut'):
	# 	if mechanism=='cut':
	# 		F = 2*beta*G*b**2*(r/rtrans)
	# 	if mechanism=='Orowan':
	# 		F = 2*beta*G*b**2

	# 	return np.sqrt(3*f_all/2/np.pi) / np.sqrt(2*beta*G*b**2)/b/r * F**1.5
	
	# r_i = np.linspace(1e-9, 8e-9, 1000)
	# tau_p_Oro = get_tau_p(r_i, rtrans=0, mechanism='Orowan')
	# tau_p_cut2 = get_tau_p(r_i, rtrans=2e-9, mechanism='cut')
	# tau_p_cut3 = get_tau_p(r_i, rtrans=3e-9, mechanism='cut')
	# tau_p_cut4 = get_tau_p(r_i, rtrans=4e-9, mechanism='cut')

	# fig, ax1 = plt.subplots(1, 1, figsize=(6,4), tight_layout=True)
	# ax1.plot(r_i/1e-9, tau_p_Oro/1e6, label='Orowan')
	# ax1.plot(r_i/1e-9, tau_p_cut2/1e6, label='cut, $r_c=2nm$')
	# ax1.plot(r_i/1e-9, tau_p_cut3/1e6, label='cut, $r_c=3nm$')
	# ax1.plot(r_i/1e-9, tau_p_cut4/1e6, label='cut, $r_c=4nm$')
	# ax1.set_xlabel('radius[nm]')
	# ax1.set_ylabel('shear stress[MPa]')
	# ax1.set_yticks(np.linspace(0, 300, 6))
	# ax1.set_ylim([0, 300])
	# ax1.grid(alpha=0.2, axis='both')


	# # discrete number density
	# Nv_i = np.zeros_like(r_i) 								# number density of each class
	# for i in range(len(r_i)):
	# 	upper_bool = prm.r<r_i[i+1] if i!=len(r_i)-1 else prm.r<-999
	# 	lower_bool = prm.r>=r_i[i]
	# 	Ni = 3*np.count_nonzero(np.logical_and(lower_bool, upper_bool)) # number of particles in class i
	# 	Nv_i[i] = Ni/prm.V

	# ax2 = ax1.twinx()
	# # ax2.hist(prm.r/1e-9, weights=prm.r/prm.r.sum(), alpha=0.4, bins=10)
	# ax2.hist(r_i/1e-9, weights=Nv_i, bins=20, alpha=0.6, density=True)
	# ax2.set_ylabel('normalized distribution[$nm^{-1}$]')
	# # ax2.set_yticks(np.linspace(0, 0.3, 6))
	# # ax2.set_ylim([0, 0.3])

	# # plt.legend()
	# plt.show()
	# >--------------------------------------------------

	return prm

def c_prec_7075(prm):
	"""
	[3] Kaka Ma, Haiming Wen, Tao Hu, Troy D. Topping, Dieter Isheim,
	David N. Seidman, Enrique J. Lavernia, Julie M. Schoenung,	
	Mechanical behavior and strengthening mechanisms in ultrafine grain
	precipitation-strengthened aluminum alloy, Acta Materialia, 2014.
	"""
	M 		= prm.M 		# (_)
	G 		= prm.G 		# (MPa)
	b 		= prm.b*1e9 	# (nm)
	nu 		= prm.nu 		# 
	rbar 	= prm.rbar 		# 
	lamda 	= prm.lamda 	# 

	########### proof ###########
	FACTOR = 1.7
	pp = FACTOR*M*0.4*G*b/(np.pi*np.sqrt(1-nu))*np.log(2*rbar/b)/lamda
	#############################

	prm.pp = pp 												# sigma p, precipitate strengthening
	return prm

def c_ss_6111(prm):
	"""
	[3] O.R Myhr, Ø Grong, S.J Andersen, Modelling of the age 
	hardening behaviour of Al-Mg-Si alloys,	Acta Materialia, 2001.
	"""
	f_all 	= prm.f_all 		# (_)
	wt 		= np.array(prm.wt) 	# (%)
	aw 		= np.array(prm.aw) 	#

	an = wt/aw 															# number of atoms
	at = an/sum(an)
	# cc_at = np.maximum(0, [at[2]-5/11*f_all, at[1]-4/11*f_all]) 		# concentration of Mg, Si (at), assuming Mg5-Al2-Si4 beta'' precipitate
	cc_at = np.maximum(0, [at[2]-0.42*f_all, at[1]-0.33*f_all]) 		# WARNING: need check !
	cc = cc_at*sum(an)*aw[[2,1]] 										# concentration of Mg, Si (wt)
	
	# print(2*aw[2]/sum([2*aw[2], 1*aw[1]]))								# Mg2-Si (wt)
	# print(5*aw[2]/sum([5*aw[2], 2*aw[0], 4*aw[1]]))						# beta'', Mg / Mg5-Al2-Si4 (wt)
	# print(9*aw[2]/sum([3*aw[0], 9*aw[2], 2*aw[3], 7*aw[1]]))			# Q, Mg / Al3-Mg9-Cu2-Si7 (wt)
	# print(5*aw[2]/sum([5*aw[2], 3*aw[1]]))							# -, Mg / Mg5-Si3 (wt)

	K = np.array([29.0, 66.3, 46.4])*1e6  								# Mg Si Cu, strength by solutes (Pa/wt%^(2/3))
	cc = np.append(cc, wt[3])
	ss = K*np.power(cc, 2/3)

	prm.cc = cc 														# concentration of solutions
	prm.ss = ss 														# sigma ss, solid solution strengthening (Pa)
	return prm

def c_ss_6111_new(prm):
	"""
	[3] O.R Myhr, Ø Grong, An Extended Age-Hardening Model for 
	Al-Mg-Si Alloys Incorporating the Room-Temperature Storage 
	and Cold Deformation Process Stages, 2015.
	"""
	f_all 	= prm.f_all 			# (_)
	wt 		= np.array(prm.wt) 	# (%)
	aw 		= np.array(prm.aw) 	#

	an = wt/aw 															# number of atoms
	at = an/sum(an)*100
	cc_at = np.maximum(0, [at[2]-0.42*f_all*100, at[1]-0.33*f_all*100]) # concentration of Mg, Si (at%)
	cc = cc_at*sum(an)/100*aw[[2,1]] 									# concentration of Mg, Si (wt%)
	K = np.array([15.0, 33.0, 46.4])*1e6  								# Mg Si Cu, strength by solutes (Pa/wt%^(2/3))
	cc = np.append(cc, wt[3])
	ss = K*np.power(cc, 2/3)

	prm.cc = cc 														# concentration of solutions
	prm.ss = ss 														# sigma ss, solid solution strengthening (Pa)
	return prm

def c_ss_7075(prm):
	"""
	Kaka Ma et, al. Mechanical behavior and strengthening mechanisms
	in ultrafine grain precipitation-strengthened aluminum alloy,
	Acta Materialia, 2014.
	"""
	f_all 	= prm.f_all 			# (_)
	wt 		= np.array(prm.wt) 	# (%)
	aw 		= np.array(prm.aw) 	#

	an = wt/aw 																	# number of atoms
	at = an/sum(an)*100
	cc_at = np.maximum(0, [at[2]-0.33*f_all*100, at[1]-0.66*f_all*100, at[3]]) 	# concentration of Mg Zn Cu (at%)
	cc = cc_at*sum(an)/100*aw[[2,1,3]] 											# concentration of Mg Zn Cu (wt%)
	K = np.array([18.6, 2.9, 13.8])*1e6  										# Mg Zn Cu, strength by solutes (Pa/wt%)
	ss = K*cc

	prm.cc = cc 												# concentration of solutions
	prm.ss = ss 												# sigma ss, solid solution strengthening
	return prm

def c_gb(prm):
	"""Hall-Petch"""
	grainSize 	= prm.grainSize 		# (m)
	ky 			= prm.ky*1e6			# (Pa*m^0.5)

	gb = ky/np.sqrt(grainSize)

	prm.gb = gb
	return prm

# >----------------------------------------------------------------------------------------------------
# > work hardening model
# >----------------------------------------------------------------------------------------------------
def c_ssd_v1(prm):
	"""
	Myhr, O.R., Grong, Ø. & Pedersen, K.O. A Combined Precipitation, Yield Strength,
	and Work Hardening Model for Al-Mg-Si Alloys. Metall Mater Trans (2010).
	"""
	T 		= prm.T 		# (K)
	b 		= prm.b 	 	# (m)
	rate 	= prm.rate 		# (s^-1)
	u0 		= prm.u0		# (Pa)
	RR 		= prm.RR 		# (kJ/mol)
	wt 		= prm.wt 		# (wt%)
	cc 		= prm.cc 		# (wt%)
	G 		= prm.G  		# (Pa)
	M 		= prm.M    		# (_)
	k1 		= prm.k1 		# (m^-1)
	k3 		= prm.k3 		# (N/m^2 wt^3/4)
	k20		= prm.k20
	Crmg	= prm.Crmg
	Zs 		= prm.Zs 		# (s^-1)
	m0 		= prm.m0		# (_)
	alpha 	= prm.alpha 	# (_)

	Cmg = np.maximum(1e-2, cc[0]+0.5*(cc[1]-0.33*(wt[4]+wt[5]))) 	# equivalent concentration of Mg (%)
	Z = rate*np.exp(68.8*1000/RR/T)

	# >--------------------------------------------------
	# naming: s1a
	k2_RT = k1*alpha*M*G*b/k3/pow(Cmg,3/4) 							# dynamic recovery rate at room temperature
	k2 = k2_RT*(1+(Zs/Z)**m0) 
	# >--------------------------------------------------
	# naming: s1b
	# k2 = k20*(G/u0)*np.power(Crmg/Cmg, 3/4)*(1+(Zs/Z)**m0) 
	prm.k2 	= k2 													# dynamic recovery rate
	return prm

def c_ssd_v2(prm):
	"""
	modified version for AA7075
	"""
	R_mean 		= prm.R_mean*1e-6 		# (mm)
	f_all  		= prm.f_all 			# (_)
	b 			= prm.b*1e3 			# (mm)
	theta0  	= prm.theta0 			# (MPa)
	M 	 		= prm.M 
	alpha   	= prm.alpha 
	G   		= prm.G 				# (MPa)
	R_trans1	= prm.R_trans1*1e-6 	# (mm)
	R_trans2	= prm.R_trans2*1e-6		# (mm)
	y0 			= prm.y0*1e-6			# (mm)
	yp 			= prm.yp*1e-6 			# (mm)

	# ####### currently no temperature consideration on k2
	# Z = rate*np.exp(68.8/RR/T*1000)
	# k2 = k2_0*G/u0*(1+(Zs/Z)**m0) 								# recovery rate considering temperature
	# #######

	k1 = 2*theta0/(M*alpha*G*b)

	L = R_mean*np.sqrt(2*np.pi/3/f_all)

	k20 = 2*y0/b
	k2p = 2*yp/b

	phi = (R_mean-R_trans1)/(R_trans2-R_trans1)

	prm.k2 	= 10 					# dummy

	prm.k1  = k1  					# (mm^-1)
	prm.L 	= L 					# (mm)
	prm.phi = np.clip(phi, 0, 1) 	# (_)
	prm.k20 = k20 					# (_)
	prm.k2p = k2p 					# (_)
	return prm

def c_gnd_v1(prm):
	"""
	Myhr, O.R., Hopperstad, O.S. & Børvik, T. A Combined Precipitation,
	Yield Stress, and Work Hardening Model for Al-Mg-Si Alloys Incorporating
	the Effects of Strain Rate and Temperature. Metall Mater Trans A, 2018.
	"""
	k1g 	= prm.k1g 			# (m^-1)
	k2g0 	= prm.k2g0 			# (_)
	lmb_star= prm.lmb_star		# (m)
	f_o 	= prm.f_o 			# (_)
	fr_o 	= prm.fr_o 			# (_)
	Zg 		= prm.Zg 			# (_)
	m0 		= prm.m0 			# 
	T 		= prm.T 			# (K)
	RR 		= prm.RR 			# (_)
	rate 	= prm.rate 			# (s^-1)

	# WARNING: K1G value is a very large value ~3e09
	K1G = k1g/lmb_star 						# dislocation accumulation term of GND
	# print(f'{K1G:10.4e}')
	# TODO: test lmb_star value

	Z = rate*np.exp(68.8*1000/RR/T)
	k2g = k2g0*(f_o/fr_o)*(1+(Zg/Z)**m0)

	prm.K1G = K1G
	prm.k2g = k2g
	return prm

def c_gnd_v2(prm):
	b 		= prm.b 	 	# (m)
	No_mean = prm.No_mean 	# (_)
	lamda 	= prm.lamda		# (m)
	f_o 	= prm.f_o 		# (_)
	ko 		= prm.ko  	 	# (_)
	ksat 	= prm.ksat 	 	# (_)

	if No_mean!=0:
		L = ko*lamda
		sat = ksat/f_o/b/L
		
		if No_mean>=30:
			a = 0.001
		elif No_mean>=20 and No_mean<30:
			a = 0.02
		elif No_mean>=10 and No_mean<20:
			a = 0.04
		else:
			a = 10
	else:
		a = 0 				# no GND effects
		L = 100   			# can be any number but 0 in Abaqus
		sat = 100			# can be any number but 0 in Abaqus

	prm.a 	= a 			# control factor
	prm.L 	= L 			# ko*lamda, (m)
	prm.sat 	= sat 		# rho g saturate, (m^-2)
	return prm

def c_wh(prm):
	alpha   = prm.alpha 		# (_)
	M 		= prm.M 			# (_)
	G 		= prm.G 			# (Pa)
	b 		= prm.b 			# (m)
	rhos 	= prm.rho_ssd_0[0]  # (m^-2)
	rhog 	= prm.rho_gnd_0[0]  # (m^-2)

	dd = alpha*M*G*b*np.sqrt(rhos+rhog) 	# work hardening by dislocation interaction

	prm.dd = dd  				# (Pa)
	return prm

# >----------------------------------------------------------------------------------------------------
# > temperature-dependent model
# >----------------------------------------------------------------------------------------------------
def c_YS_temper(prm, T=None):
	T 		= T or prm.T 		# (K)
	rate 	= prm.rate 			# (s^-1)
	u0 		= prm.u0 			# (Pa)
	pp 		= prm.pp			# (Pa)
	ss 		= prm.ss 			# (Pa)
	gb 		= prm.gb 			# (Pa)
	G 		= prm.G 			# (Pa)
	c1 		= prm.c1  			# (_)
	RR 		= prm.RR 	 		# (kJ/mol)
	DG 		= prm.DG 			# (J)
	e0 		= prm.e0 			# (s^-1)
	M 		= prm.M 			# (_)
	q 		= prm.q  			# (_), 1~2
	p 		= prm.p  			# (_), 0~1
	
	YS = 10e6+sum(ss)+gb+pp													# YS in room temperature
	YS_temper = YS/c1*G/u0* \
				np.power(1-np.power(T*RR/DG*np.log(e0/rate), 1/q), 1/p) 	# YS considering temperature
	tau = YS_temper/M   													# Taylor factor M = 2.4, proof required
	prm.YS = YS_temper 														# sigma y, yield stress
	prm.tau 	= tau 														# CRSS
	prm.xi_0_sl = tau
	return prm



