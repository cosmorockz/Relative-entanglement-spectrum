import numpy as np
from mpmath import *
from LL_wf import SLL

def overlap_integrand(theta, Q, l1, l2, m):
    wf1 = SLL(theta,0,Q,l1,m)
    wf2 = SLL(theta,0,Q,l2,m)
    x = 2 * pi * sin(theta) *( wf1 * wf2 )
    return x

def overlap_integral(Q, l1, l2, m, theta_cut):
    x = quad(lambda theta: overlap_integrand(
        theta, Q, l1, l2, m), [0, theta_cut])
    return x

