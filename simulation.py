from __future__ import division

import numpy as np
import random,os 
import matplotlib.pyplot as plt
import utils as tech

from matplotlib import rcParams
from scipy.stats import scoreatpercentile

rcParams['text.usetex'] = True

'''
MUST VECTORIZE THIS CODE AND SPLIT IT INTO MULTIPLE FILES
'''

def viral_bases(eigenvalues,eigenvectors):
	return [(w,v) for w,v in zip(eigenvalues,eigenvectors) if w.real > 1]

def popularity(activity):
	activity[activity==-1]=0
	return activity.sum(axis=0)/float(activity.shape[0])

def activity_wavform_analysis(activity):
	viral_threshold = scoreatpercentile(activity,85)
	return np.where(activity>viral_threshold)[0]

def partition_data(data,condition_duration,experiment_duration):
	segments =experiment_duration//condition_duration
	return [[datum for datum in data 
			if datum > (n*condition_duration) and datum < ((n+1)*condition_duration)]
			 for n in range(segments)]

def iqr(data,axis=0):
	return 0.5*(np.percentile(data,75,axis=axis)-np.percentile(data,25,axis=axis))

def make_connection_matrix(n,mode='random',influencers=0.1):
	if mode == 'random':
		M = 2*np.random.random_sample(size=(n,n))-1
		M[np.diag_indices_from(M)] = 0
		return M
	elif mode == 'smallworld':
		#Influencers have a degress defined by a uniform distribution
		#Other have have degrees defined by tight Gaussian
		influencerM = np.random.random_sample(size=(int(influencers*n),n))
		restM = np.random.randn(n-int(influencers*n),n)
		M = np.r_[influencerM,restM]
		return M
	else:
		print 'Mode not understood'
sgn = lambda x: 1 if x > 0 else -1 

n = 1000

INITIAL = 0
timesteps = 300

boltzmann = lambda activity: 2/(1+np.exp(-activity))-1
epsilon = 0.001

repeats = 20 
DATA = 'data'
for mode in ['random','smallworld']:
	base_savepath = os.path.join(os.getcwd(),DATA,mode)
	database = {}
	for repeat in xrange(repeats):
		#--initial conditions
		M = make_connection_matrix(n,mode=mode)
		eigenvalues,eigenvectors = np.linalg.eig(M)

		a_initial = np.random.random_sample(size=(n,1))
		a = np.tile(a_initial,(1,4*timesteps)).astype(np.complex64)
		for t in xrange(1,timesteps):
			update = random.choice(range(n))
			a[update,t] = 1 if boltzmann(M[update,:].dot(a[:,t-1])) > np.random.random() else -1 

		viral_support= viral_bases(eigenvalues,eigenvectors)
		important_values,important_vectors = zip(*viral_support)
		important_vectors = np.array(important_vectors)
		important_values = np.array(important_values)

		stimulus = [(vector,weight) for vector,weight in zip(important_vectors, important_values) 
				if weight > scoreatpercentile(important_values.real,85) and 
				len(vector[vector>0]/float(len(vector))) > scoreatpercentile([len(eigenvector[eigenvector>0])/float(len(eigenvector)) 
					for eigenvector in important_vectors],85)]

		VECTOR = 0
		WEIGHT = 1
		if len(stimulus) ==0:
			stimulus = 0
		elif len(stimulus) ==1:
			stimulus = stimulus[0][VECTOR]
		elif len(stimulus) > 1:
			MSE = [len(vector[vector>1])/float(len(vector)) + weight/float(sum(x[WEIGHT] for x in stimulus))
				for vector,weight in stimulus]
			stimulus = stimulus[np.argmax(MSE)][VECTOR]

		#--The above few lines determine WHICH agents to target, nothing in this model specifies the mechanism 
		#--of stimulating interest. 

		stimulus[stimulus.real>0] = 1
		stimulus[stimulus!=1] = 0
		stimulus = stimulus.astype(int)
		#-- Here zero denotes not stimulating, not sure how realistic inhibiting nodes in a social network is in 
		#-- this model

		#stimulus = important_vectors[np.argmax(important_values)]
		alpha = .5

		for t in xrange(timesteps,2*timesteps):
			#update = random.choice(range(n))
			penetrance = 0.1
			for update in random.sample(xrange(n),int(penetrance*n)):
				a[update,t] = stimulus[update] if stimulus[update] ==1 else sgn(boltzmann(M[update,:].dot(a[:,t-1]))-np.random.random())
			#a[update,t] = stimulus[update]

		for t in xrange(2*timesteps,3*timesteps):
			update = random.choice(range(n))
			a[update,t] = 1 if boltzmann(M[update,:].dot(a[:,t-1])) > np.random.random() else -1 

		for t in xrange(3*timesteps,4*timesteps):
			update = random.choice(range(n))
			a[update,t] = 1 if np.random.random() > 0.5 else -1 
		

		timestamps = activity_wavform_analysis(popularity(a))
		database[str(repeat)] = {'activity':a,'connections':M,'timestamps':timestamps, 
				'stimulus': 0 if stimulus is 0 else 1}
		coefficients = important_vectors.dot(a) 
		#--Dynamics
		fig,axs = plt.subplots(nrows=2,ncols=1)
		#axs[0].errorbar(range(timesteps),a.mean(axis=0),yerr=a.std(axis=0),fmt='o',ecolor='k',capthick=2)
		axs[0].plot(popularity(a),'k',linewidth=2, clip_on=False)
		axs[0].axvline(x=timesteps,c='r',linestyle='--',linewidth=2,clip_on=False)
		axs[0].axvline(x=2*timesteps,c='r',linestyle='--',linewidth=2,clip_on=False)
		axs[0].axvline(x=3*timesteps,c='r',linestyle='--',linewidth=2,clip_on=False)
		#axs[0].plot(timestamps,popularity(a)[timestamps]+0.04,"rv",clip_on=False)
		axs[0].axhline(scoreatpercentile(popularity(a),85),c='k',linestyle='--',linewidth=2)
		normalized_coefficients = coefficients.real.astype(np.float32)
		normalized_coefficients = normalized_coefficients/normalized_coefficients.max(axis=1)[:,np.newaxis]
		normalized_coefficients.sort(axis=0)
		cax =axs[1].imshow(normalized_coefficients[::-1][:10],interpolation='nearest',aspect='auto',cmap=plt.cm.seismic)
		map(tech.adjust_spines,axs)
		axs[0].set_ylabel(r'\Large $\mathbf{a\left(t\right)}$')
		axs[1].set_xlabel(r'\Large \textsc{Time}')
		axs[1].set_ylabel(r'\Large $\mathbf{c_n \cdot a\left(t\right)}$')
		plt.savefig(os.path.join(base_savepath,'network-activity-repeat-%d.png'%repeat),dpi=300)
		#--Topology
		
		connections = plt.figure()
		panel = connections.add_subplot(111)
		cax = panel.imshow(M,aspect='equal',interpolation='nearest',cmap=plt.cm.seismic)
		plt.colorbar(cax)
		tech.adjust_spines(panel)
		panel.set_xlabel(r'\Large \textsc{From}')
		panel.set_ylabel(r'\Large \textsc{To}')
		plt.savefig(os.path.join(base_savepath,'connections-%d'%repeat),dpi=300)
		#--Eigenanalysis
		
		eigenfigure,(eig_panel_vectors,eig_panel_values) = plt.subplots(nrows=2,ncols=1)
		eig_panel_vectors.imshow(np.absolute(important_vectors.T),interpolation='nearest',aspect='auto',cmap=plt.cm.seismic)
		eig_panel_values.scatter(eigenvalues.imag,eigenvalues.real,s=30,facecolor='1',edgecolor='k')
		eig_panel_values.scatter(important_values.imag,important_values.real,s=30,facecolor='0',edgecolor='k')
		#eig_panel_values.stem(np.absolute(important_values)/np.absolute(eigenvalues).sum(),markerfmt='ko',basefmt='k--')
		tech.adjust_spines(eig_panel_vectors)
		tech.adjust_spines(eig_panel_values)

		eig_panel_vectors.set_ylabel(r'\Large \textbf{\textsc{Agent}}')
		eig_panel_values.set_ylabel(r'\Large \textbf{\textsc{Real}}')
		eig_panel_values.set_xlabel(r'\Large \textbf{\textsc{Imaginary}}')
		plt.savefig(os.path.join(base_savepath,'eigenanalysis-repeat-%d.png'%repeat),dpi=300)
		#--Efficacy
		counts = map(len,partition_data(timestamps,timesteps,3*timesteps))

		eff = plt.figure()
		eff_ax = eff.add_subplot(111)
		width=0.2
		#to get error bars need to do trials
		eff_ax.bar(0.5*np.arange(len(counts)),counts,width,color='black', clip_on=False)
		tech.adjust_spines(eff_ax)
		eff_ax.set_xticks(0.5*np.arange(len(counts))+.1)
		eff_ax.set_xticklabels([r'\Large \textsc{Before}',r'\Large \textsc{During}',r'\Large \textsc{After}'])
		eff_ax.set_ylabel(r'\Large \textsc{Count of viral events}')
		plt.savefig('efficacy-repeat-%d.png'%repeat,dpi=300)

		ranking_eig_vecs = plt.figure()
		r_ax = ranking_eig_vecs.add_subplot(111)
		x =  [len(eigenvector[eigenvector>0])/float(len(eigenvector)) for eigenvector in important_vectors]
		y = important_values.real
		x_threshold = scoreatpercentile(x,85)
		y_threshold = scoreatpercentile(y,85)
		r_ax.scatter(x,y,facecolor='0',edgecolor='k')
		r_ax.axvline(x_threshold,c='r',linewidth=2,linestyle='--')
		r_ax.axhline(y_threshold,c='r',linewidth=2,linestyle='--')
		with open(os.path.join(base_savepath,'number-of-viral-supports'),'a') as f:
			print>>f,len([(x,y) for x,y in zip(x,y) if x>x_threshold and y>y_threshold])
		tech.adjust_spines(r_ax)
		r_ax.set_ylabel(r'\Large $\Re(\lambda)$')
		r_ax.set_xlabel(r'\Large \textbf{\textsc{No. of interested agents}}')
		plt.savefig(os.path.join(base_savepath,'stimulus-discovery-%d.png'%repeat),dpi=300)
		plt.close()

	#--Overall efficacy
	counts = np.array([map(len,partition_data(database[repeat]['timestamps'],timesteps,3*timesteps)) 
		for repeat in database if database[repeat]['stimulus']==1])


	full_efficacy_figure = plt.figure()
	full_efficacy_axis = full_efficacy_figure.add_subplot(111)
	width=0.2

	iqrs = iqr(counts,axis=0)
	full_efficacy_axis.bar(0.5*np.arange(counts.shape[1])-0.1,counts.mean(axis=0)-iqrs,width, color='black',
			yerr=2*iqrs,error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),clip_on=False)
	tech.adjust_spines(full_efficacy_axis)
	full_efficacy_axis.set_xticks(0.5*np.arange(counts.shape[1]))
	full_efficacy_axis.set_xticklabels([r'\Large \textsc{Before}',r'\Large \textsc{During}',r'\Large \textsc{After}'])
	full_efficacy_axis.set_ylabel(r'\Large \textsc{Count of viral events}')
	plt.savefig(os.path.join(base_savepath,'summary-efficacy-repeat.png'),dpi=300)