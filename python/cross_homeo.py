#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-unit rate-based population model with cross-homeostatic learning rules
S. Soldado-Magraner, R. Laje, D.V. Buonomano (2022)

@author: S. Soldado-Magraner
@author: R. Laje

"""

#%% Libraries

import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from IPython import get_ipython
# from cross_homeo_graphics import plot_realtime

get_ipython().run_line_magic('matplotlib', 'qt') # plot in a separate window
# get_ipython().run_line_magic('matplotlib', 'inline') # plot inline



def plot_realtime():

	fig, axs = plt.subplots(5,2,num=1,clear=True,figsize=(4,8))
	(ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10) = axs
	gs = axs[4, 0].get_gridspec()
	for ax in axs[4,:]:
		ax.remove()
	ax_meanEI = fig.add_subplot(gs[4,:])
	gs = axs[0, 1].get_gridspec()
	for ax in axs[0:2,1]:
		ax.remove()
	ax_WEE_WEI = fig.add_subplot(gs[0:2,1])
	gs = axs[2, 1].get_gridspec()
	for ax in axs[2:4,1]:
		ax.remove()
	ax_WIE_WII = fig.add_subplot(gs[2:4,1])
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
 	
	fsize = 8
	fsize2 = 10
	E_color = (0, 0.5, 0)
	I_color = (1, 0, 0)
	FCaExc_color = (77/255, 166/255, 77/255, 0.25)
	WEEpresyn_color = (196/255, 193/255, 193/255)
	lwidth = 1
	msize = 2


	fig.suptitle('Training',fontsize=fsize2,fontweight='bold')

	time_axis = np.arange(1-EvokedOn, t_max-EvokedOn+1, 1)*dt
	trial_axis = np.arange(1, n_trials, 1)

	ax1.plot(time_axis,hR[:,18], color=E_color, linewidth=lwidth)
	ax1.hlines(E_set, time_axis[0], time_axis[-1], colors=E_color, linewidth=lwidth, linestyles='dotted')
	ax1.set_xlim(dt-EvokedOn*dt, 0.7)
	ax1.set_ylim(0, 20)
	ax1.set_ylabel('E (Hz)', fontsize=fsize)
	ax1.tick_params(axis='x',labelsize=fsize)
	ax1.tick_params(axis='y',labelsize=fsize)
 
	ax3.plot(time_axis,hR[:,40], color=E_color, linewidth=lwidth)
	ax3.hlines(E_set, time_axis[0], time_axis[-1], colors=E_color, linewidth=lwidth, linestyles='dotted')
	ax3.set_xlim(dt-EvokedOn*dt, 0.7)
	ax3.set_ylim(0, 20)
	ax3.set_ylabel('E (Hz)', fontsize=fsize)
	ax3.tick_params(axis='x',labelsize=fsize)
	ax3.tick_params(axis='y',labelsize=fsize)
 
	ax5.plot(time_axis,hR[:,N_E+1], color=I_color, linewidth=lwidth)
	ax5.hlines(E_set, time_axis[0], time_axis[-1], colors=I_color, linewidth=lwidth, linestyles='dotted')
	ax5.set_xlim(dt-EvokedOn*dt, 0.7)
	ax5.set_ylim(0, 55)
	ax5.set_ylabel('I (Hz)', fontsize=fsize)
	ax5.tick_params(axis='x',labelsize=fsize)
	ax5.tick_params(axis='y',labelsize=fsize)
 
	ax7.plot(time_axis,hR[:,N_E+12], color=I_color, linewidth=lwidth)
	ax7.hlines(E_set, time_axis[0], time_axis[-1], colors=I_color, linewidth=lwidth, linestyles='dotted')
	ax7.set_xlim(dt-EvokedOn*dt, 0.7)
	ax7.set_ylim(0, 55)
	ax7.set_xlabel('Time (s)', fontsize=fsize)
	ax7.set_ylabel('I (Hz)', fontsize=fsize)
	ax7.tick_params(axis='x',labelsize=fsize)
	ax7.tick_params(axis='y',labelsize=fsize)

# 	ax_meanEI.plot(npm.repmat(trial_axis,1,1), trialhist_FCaExc[:,0:20], color=FCaExc_color)
# 	ax_meanEI.plot(trial_axis, np.mean(trialhist_FCaExc, axis=1), color=FCaExc_color)
# 	ax_meanEI.set_xlabel('Trial', fontsize=fsize)
# 	ax_meanEI.set_ylabel('Mean E/I (Hz)', fontsize=fsize)
# 	ax_meanEI.tick_params(axis='x',labelsize=fsize)
# 	ax_meanEI.tick_params(axis='y',labelsize=fsize)


	ax_WEE_WEI.plot(trialhist_WEEpresyn, trialhist_WEIpresyn, 'o-', color=WEEpresyn_color, linewidth=lwidth, markersize=msize)
	ax_WEE_WEI.plot(W_EEpresyn, W_EIpresyn, 'o', color=E_color, linewidth=lwidth, markersize=msize)
	ax_WEE_WEI.set_xlabel('WEE', fontsize=fsize)
	ax_WEE_WEI.set_ylabel('WEI', fontsize=fsize)
	ax_WEE_WEI.tick_params(axis='x',labelsize=fsize)
	ax_WEE_WEI.tick_params(axis='y',labelsize=fsize)

	ax_WIE_WII.plot(trialhist_WIEpresyn, trialhist_WIIpresyn, 'o-', color=WEEpresyn_color, linewidth=lwidth, markersize=msize)
	ax_WIE_WII.plot(W_IEpresyn, W_IIpresyn, 'o', color=I_color, linewidth=lwidth, markersize=msize)
	ax_WIE_WII.set_xlabel('WIE', fontsize=fsize)
	ax_WIE_WII.set_ylabel('WII', fontsize=fsize)
	ax_WIE_WII.tick_params(axis='x',labelsize=fsize)
	ax_WIE_WII.tick_params(axis='y',labelsize=fsize)


	fig.set_size_inches(8, 8, forward=True)
	plt.savefig('prueba.pdf')

# 	plt.close(fig)
	return fig



#%% Numerics

dt = 0.0001 # in seconds
t_max = 20000 #2/dt # duration of a trial
n_trials = 100 #200 # number of trials

GRAPHICS = 1
VIDEO = 0
HOMEOSTATIC_FLAG = 1



#%% Neuronal parameters

# activation function (ReLU)
def activ_f(x, gain, threshold):
	return gain*np.maximum(0, x - threshold)

N_E = 80 # number of Exc units
N_I = 20 # number of Inh units

theta_E = 4.8
theta_I = 25
gain_E = 1
gain_I = 4

tau_E = 10/(dt*1000)
tau_I = 2/(dt*1000)

beta = 0
tau_A = 500

E_max = 100
I_max = 250

OU_tau = 0.1 # Ornstein-Uhlenbeck noise
OU_mu = 0;
OU_sigma = 0.1 # sigma * sqrt(dt)



#%% Plasticity parameters

E_set = 5 # setpoint for excitatory neurons
I_set = 14 # setpoint for inhibitory neurons

tau_trial = 2

W_EI_min = 0.1
W_EE_min = 0.1
W_II_min = 0.1
W_IE_min = 0.1

alpha = 0.00002 # learning rate 



#%% Matrix initialization

W_EE = np.random.rand(N_E, N_E)*0.16 # why uniform instead of gaussian?
W_EE = W_EE - np.diag(np.diag(W_EE)) # why removing self-connections?
W_EI = np.random.rand(N_E, N_I)*0.16
W_IE = np.random.rand(N_I, N_E)*0.16
W_II = np.random.rand(N_I, N_I)*0.16
W_II = W_II - np.diag(np.diag(W_II))

W_init = np.vstack((np.hstack((W_EE, W_EI)), np.hstack((W_IE, W_II))))


#%% Variable initialization

trialhist_FCaExc = np.full((n_trials, N_E), np.nan)
trialhist_FCaInh = np.full((n_trials, N_I), np.nan)
trialhist_WEE = np.full((n_trials, N_E), np.nan)
trialhist_WEI = np.full((n_trials, N_E), np.nan)
trialhist_WIE = np.full((n_trials, N_I), np.nan)
trialhist_WII = np.full((n_trials, N_I), np.nan)
trialhist_WEEpresyn = np.full((n_trials, N_E), np.nan)
trialhist_WEIpresyn = np.full((n_trials, N_E), np.nan)
trialhist_WIEpresyn = np.full((n_trials, N_I), np.nan)
trialhist_WIIpresyn = np.full((n_trials, N_I), np.nan)

W_EEpresyn = np.sum(W_EE, axis=1) # update sum of presynaptic weights for plot
W_EIpresyn = np.sum(W_EI, axis=1)
W_IEpresyn = np.sum(W_IE, axis=1)
W_IIpresyn = np.sum(W_II, axis=1)

E_avg = np.zeros(N_E)
I_avg = np.zeros(N_I)

hR = np.zeros((t_max, N_E+N_I)) # history of Inh and Exc rate

EvokedOn = round(0.250/dt) # start of Evoked current
EvokedDur = round(0.01/dt)
EvokedAmp = 7

counter = 0

OU_E = np.zeros(N_E)
OU_I = np.zeros(N_I)



#%% Simulation

if GRAPHICS:
	fig_training = plot_realtime()
	fig_training

for trial in range(n_trials):
	if trial%10==0 or trial==n_trials-1:
		print('trial=' + str(trial))
# 	f_Ca_Exc = np.zeros((N_E, t_max)) # instantaneous fast Ca sensor, integrates the firing rate of E
# 	f_Ca_Inh = np.zeros((N_I, t_max)) # instantaneous fast Ca sensor, integrates the firing rate of I
	f_Ca_Exc = np.zeros((t_max, N_E)) # instantaneous fast Ca sensor, integrates the firing rate of E
	f_Ca_Inh = np.zeros((t_max, N_I)) # instantaneous fast Ca sensor, integrates the firing rate of I

	hR = np.zeros((t_max, N_E+N_I)) #history of Inh and Exc rates

	E = np.zeros(N_E)
	I = np.zeros(N_I)
	a = np.zeros(N_E)

	evoked = np.zeros(t_max)
	evoked[EvokedOn:EvokedOn+EvokedDur] = EvokedAmp

	for t in range(t_max):
		# Ornstein-Uhlenbeck noise
		OU_E = OU_E + OU_tau*(OU_mu - OU_E) + OU_sigma*np.random.randn(N_E)
		OU_I = OU_I + OU_tau*(OU_mu - OU_I) + OU_sigma*np.random.randn(N_I)

		# neuronal dynamics (Euler method)
		E = E + (-E + activ_f(W_EE@E - W_EI@I - a + evoked[t] + OU_E, gain_E, theta_E) )/tau_E
		I = I + (-I + activ_f(W_IE@E - W_II@I + OU_I, gain_I, theta_I) )/tau_I

		# adaptation
		a = a + (-a + beta*E)/tau_A

		E_maxvec = E>E_max; E[E_maxvec] = E_max # neurons have a saturation of their rates
		I_maxvec = I>I_max; I[I_maxvec] = I_max

		hR[t,] = np.hstack((E, I))

		# Calcium sensors
		f_Ca_Exc[t,:] = E
		f_Ca_Inh[t,:] = I

	E_avg = E_avg + (-E_avg + np.mean(f_Ca_Exc[-round(0.5/dt):,:], axis=0))/tau_trial # we average at the end of the trial to avoid evoked
	I_avg = I_avg + (-I_avg + np.mean(f_Ca_Inh[-round(0.5/dt):,:], axis=0))/tau_trial
      
	W_EEpresyn = np.sum(W_EE, axis=1) # update sum of presynaptic weights for plot
	W_EIpresyn = np.sum(W_EI, axis=1)
	W_IEpresyn = np.sum(W_IE, axis=1)
	W_IIpresyn = np.sum(W_II, axis=1)

	trialhist_FCaExc[trial,:] = E_avg
	trialhist_FCaInh[trial,:] = I_avg
	trialhist_WEE[trial,:] = W_EE[:,-1]
	trialhist_WEI[trial,:] = W_EI[:,-1]
	trialhist_WIE[trial,:] = W_IE[:,-1]
	trialhist_WII[trial,:] = W_II[:,-1]
	trialhist_WEEpresyn[trial,:] = W_EEpresyn
	trialhist_WEIpresyn[trial,:] = W_EIpresyn
	trialhist_WIEpresyn[trial,:] = W_IEpresyn
	trialhist_WIIpresyn[trial,:] = W_IIpresyn


      # x=WEEp;
      # y1=WEIp;
      # [R,p] = corr(x,y1,'rows','complete');
      # P = polyfit(x,y1,1);
      # yfit = P(1)*x+P(2);ix=WIEp;
      # iy1=WIIp;
      # [iR,ip] = corr(ix,iy1,'rows','complete');
      # iP = polyfit(ix,iy1,1);
      # iyfit = iP(1)*ix+iP(2);

	if HOMEOSTATIC_FLAG:
		# average activity is rectified for trials that start with 0 rate (development settings),
		# otherwise weights would never move
		E_avg = np.maximum(1,E_avg)
		I_avg = np.maximum(1,I_avg)

		# learning rule
		W_EE = W_EE + alpha*E_avg*np.sum(I_set - I_avg)/N_I
		W_EE = W_EE - np.diag(np.diag(W_EE))
		W_EI = W_EI - alpha*I_avg*np.sum(I_set - I_avg)/N_I
		W_IE = W_IE - alpha*E_avg*np.sum(E_set - E_avg)/N_E
		W_II = W_II + alpha*I_avg*np.sum(E_set - E_avg)/N_E
		W_II = W_II - np.diag(np.diag(W_II))


		# If weights fall below a minimum or are NaN set to minimum
		W_EE[W_EE < W_EE_min/(N_E-1)] = W_EE_min/(N_E-1)
# 		W_EE(isnan(WEE))=WEE_MIN/(Ne-1);
		W_EI[W_EI < W_EI_min/N_I] = W_EI_min/N_I
        # WEI(isnan(WEI))=WEI_MIN/Ni;
		W_IE[W_IE < W_IE_min/N_E] = W_IE_min/N_E
        # WIE(isnan(WIE))=WIE_MIN/Ne;
		W_II[W_II < W_II_min/(N_I-1)] = W_II_min/(N_I-1)
        # WII(isnan(WII))=WII_MIN/(Ni-1);
		W_EE = W_EE - np.diag(np.diag(W_EE))
		W_II = W_II - np.diag(np.diag(W_II))

	if GRAPHICS & trial%10==0:
		fig_training.canvas.draw()
# 		fig_training.flush_events()

# %           if ismember(trial,savetrials)
# %               saveas(gcf,['trial',num2str(trial)],'jpg')
# %               saveas(gcf,['trial',num2str(trial)],'svg')
# % 
# %           end

# 	if VIDEO:
# 		counter = counter + 1
# 		frames(counter) = getframe(h1)
   

W_end = np.vstack((np.hstack((W_EE, W_EI)), np.hstack((W_IE, W_II))))



#%% Plotting


fig, axs = plt.subplots(1,2,num=2,clear=True,figsize=(4,4))
ax1, ax2 = axs
plt.tight_layout()
plt.subplots_adjust(top=0.9)

fsize = 8
fsize2 = 10
E_color = (0, 0.5, 0)
I_color = (1, 0, 0)
FCaExc_color = (77/255, 166/255, 77/255, 0.25)
WEEpresyn_color = (196/255, 193/255, 193/255)
lwidth = 1
msize = 2


fig.suptitle('Weights',fontsize=fsize2,fontweight='bold')

img = ax1.imshow(W_init)
ax1.set_xlabel('presyn', fontsize=fsize)
ax1.set_ylabel('postsyn', fontsize=fsize)
ax1.set_title('Pre-training',fontsize=fsize2,fontweight='bold')
fig.colorbar(img, ax=ax1, shrink=0.5)

ax2.imshow(W_end)
ax2.set_xlabel('presyn', fontsize=fsize)
ax2.set_ylabel('postsyn', fontsize=fsize)
ax2.set_title('Post-training',fontsize=fsize2,fontweight='bold')
fig.colorbar(img, ax=ax2, shrink=0.5)


fig.set_size_inches(8, 8, forward=True)
plt.savefig('prueba.pdf')




fig, axs = plt.subplots(2,2,num=3,clear=True,figsize=(4,4))
(ax11, ax12), (ax21, ax22) = axs
plt.tight_layout()
plt.subplots_adjust(top=0.9)

ax11.hist(W_init[0:N_E,0:N_E])
ax11.hist(W_EE)
ax11.set_xlabel('WEE', fontsize=fsize)

ax12.hist(W_init[0:N_E,N_E:-1])
ax12.hist(W_EI)
ax12.set_xlabel('WEI', fontsize=fsize)

ax21.hist(W_init[N_E:-1,0:N_E])
ax21.hist(W_IE)
ax21.set_xlabel('WIE', fontsize=fsize)

ax22.hist(W_init[N_E:-1,N_E:-1])
ax22.hist(W_II)
ax22.set_xlabel('WII', fontsize=fsize)



#%%



