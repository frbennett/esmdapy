import numpy as np


###################################################################################################
# Routines for parameter transform
###################################################################################################
def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def inverse_scale_param(P, a, b):
    p_range = b - a
    param = P  * p_range + a
    return param

def transform(P, a, b):
    in_log = inv_logit(P)
#    parameter = inverse_transform_param(in_log, a, b)
    return parameter

def scale_param(P, a, b):
    return (P-a)/(b-a)

