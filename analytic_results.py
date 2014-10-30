import numpy as np
import utils as tech
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['text.usetex'] = True

duration = 50
n = 2

interest = np.zeros((n,duration))
Wu = np.array([1,1])

b = 1
a = 1
M  = np.array([[0,b],[a,0]])

eigvals,eigvecs = np.linalg.eig(M)

coefficients = np.zeros((eigvecs.shape[0],duration),dtype=complex)
interest[:,0] = np.random.random_sample((n,))
interest_inf = 2*np.array([float(a/b),1])
coefficients[:,0] = eigvecs.T.dot(np.conj(interest[:,0])) #Numpy dot doesn't take the complex conjugate

norm_factos = np.sum(eigvecs,axis=0)/(1.-eigvals)
for t in range(1,duration):
	interest[:,t] = interest_inf + (interest[:,0]-interest_inf)*np.exp(-t)
	coefficients[:,t] = norm_factos*(1-np.exp(-t*(1-eigvals))) + coefficients[:,0]*np.exp(-t*(1-eigvals))

phase_plot = plt.subplot(121)
phase_plot.plot(interest[1,:],interest[0,:],'k.',linewidth=2,clip_on=False)
#-- plot nullclines
phase_plot.plot(interest[1,:],b*interest[0,:] + 1,'k-',linewidth=2,clip_on=False,label=r'\Large $\mathbf{a_1}$')
phase_plot.plot(interest[0,:],a*interest[1,:] + 1,'k--',linewidth=2,clip_on=False,label=r'\Large $\mathbf{a_0}$')
tech.adjust_spines(phase_plot)
phase_plot.set_ylabel(r'\Large $\mathbf{a_0}$',rotation='horizontal')
phase_plot.set_xlabel(r'\Large $\mathbf{a_1}$')
plt.legend(frameon=False)

eigenplane = plt.subplot(122)
eigenplane.scatter(coefficients[1,:].real,coefficients[1,:].imag,
	facecolors='none',edgecolors='k', label=r'\Large $c_1$')
eigenplane.scatter(coefficients[0,:].real,coefficients[0,:].imag,
	facecolors='k',edgecolors='k', label=r'\Large $c_0$')
tech.adjust_spines(eigenplane)
eigenplane.set_xlabel(r'\Large \textbf{\textsc{Real}}')
eigenplane.set_ylabel(r'\Large \textbf{\textsc{Imaginary}}')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

timeseries,(RE,IM) = plt.subplots(nrows=2,ncols=1,sharex=True)
RE.plot(coefficients[1,:].real,'k-',linewidth=2,label=r'\Large $c_1$')
IM.plot(coefficients[1,:].imag,'k-',linewidth=2,label=r'\Large $c_1$')
RE.plot(coefficients[0,:].real,'k--',linewidth=2,label=r'\Large $c_0$')
IM.plot(coefficients[0,:].imag,'k--',linewidth=2,label=r'\Large $c_0$')
tech.adjust_spines(RE)
tech.adjust_spines(IM)
IM.set_xlabel(r'\Large \textbf{\textsc{Time}}')
IM.set_ylabel(r'\Large \textbf{\textsc{Imaginary}}')
RE.set_ylabel(r'\Large \textbf{\textsc{Real}}')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()