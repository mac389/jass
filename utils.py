from __future__ import division
import numpy as np
from scipy.linalg import circulant

def adjust_spines(ax, spines=['bottom','left']):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def _distance_matrix(L):
    Dmax = L//2
 
    D  = range(Dmax+1)
    D += D[-2+(L%2):0:-1]
 
    return circulant(D)/Dmax
 
def _pd(d, p0, beta):
    return beta*p0 + (d <= p0)*(1-beta)
 
def watts_strogatz(L, p0, beta, directed=False, rngseed=1):
    """
    Watts-Strogatz model of a small-world network
 
    This generates the full adjacency matrix, which is not a good way to store
    things if the network is sparse.
 
    Parameters
    ----------
    L        : int
               Number of nodes.
 
    p0       : float
               Edge density. If K is the average degree then p0 = K/(L-1).
               For directed networks "degree" means out- or in-degree.
 
    beta     : float
               "Rewiring probability."
 
    directed : bool
               Whether the network is directed or undirected.
 
    rngseed  : int
               Seed for the random number generator.
 
    Returns
    -------
    A        : (L, L) array
               Adjacency matrix of a WS (potentially) small-world network.
 
    """
    np.random.seed(rngseed)
 
    d = _distance_matrix(L)
    p = _pd(d, p0, beta)
 
    if directed:
        A = 1*(np.random.random(p.shape) < p)
        np.fill_diagonal(A, 0)
    else:
        upper = np.triu_indices(L, 1)
 
        A          = np.zeros(p.shape, dtype=int)
        A[upper]   = 1*(np.random.random(len(upper[0])) < p[upper])
        A.T[upper] = A[upper]
 
    return A