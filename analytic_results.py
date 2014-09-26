import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['text.usetex'] = True

duration = 20
n = 2

interest = np.zeros((n,duration))
Wu = np.array([1,1])

b = -1
a = -1
M  = np.array([[0,b],[a,0]])

interest[:,0] = np.random.random_sample((n,))
interest_inf = 1./(1+a*b)*np.array([1.+a, 1.+b])
for t in range(1,duration):
	interest[:,t] = interest_inf + (interest[:,0]-interest_inf)*np.exp(-t)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(interest[1,:],interest[0,:])
plt.show()