import numpy as np
import scipy.linalg as sla

def power_iteration(A, Omega, power_iter = 3):
    Y = A @ Omega
    for q in range(power_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q

def rtsvd(A, rank, power_iter = 3):
    Omega = np.random.randn(A.shape[1], rank+1)
    Q = power_iteration(A, Omega, power_iter)
    B = Q.T @ A
    u_tilde, s, v = np.linalg.svd(B, full_matrices = 0)
    u = Q @ u_tilde
    value = s.sum()/(s.sum()+((A.shape[1]-rank)*s.min()))
    u = u[:, :rank]
    v = v[:rank]
    s = s[:rank]
    return u, s, v, value

def tsvd(a,rank):
    u, s, v = np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=True)
    total = s.sum()
    u = u[:, :rank]
    v = v[:rank]
    s = s[:rank]
    value = s.sum()/total
    return u, s, v, value

def tinv(a, rank, type='svd', power_iter=3):
    print('Inverse type set to ', type)

    if type == 'tsvd':
        u, s, v, value = tsvd(a, rank)
        pinverse = v.T @ np.diag(s**-1) @ u.T
        print('Approx variance recovered after truncation ', value)
    if type == 'rtsvd':
        u, s, v, value = rtsvd(a, rank, power_iter = power_iter)
        pinverse = v.T @ np.diag(s**-1) @ u.T
        print('Approx variance recovered after rtsvd truncation ', value)
    if type == 'svd':
        pinverse, svd_rank = sla.pinvh(a, return_rank=True)
        print('Rank from full SVD = ', svd_rank)
    
    return pinverse 

def pseudo_inverse(del_D, alpha, Cd, nEnsemble, dLength, mLength, type='svd'):
    if type == 'svd':
        Cdd = (del_D@del_D.T)/(nEnsemble-1)
        K = Cdd + alpha*Cd
        Kinv, svd_rank = sla.pinvh(K, return_rank=True)
    #    print('Rank : ', svd_rank)
    return Kinv



