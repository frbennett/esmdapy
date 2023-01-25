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


def sub_space_inversion(Delta_D, F, rank):
    Ud, Wd, Vd = np.linalg.svd(Delta_D, full_matrices=True, compute_uv=True, hermitian=False)
    Ud = Ud[:, :rank]
    Vd = Vd[:rank]
    Wd = Wd[:rank]
    r = Wd.size
    Ir = np.diag(np.ones(r))
    X = (np.diag(Wd**-1)@Ud.T@F@Ud@np.diag(Wd**-1))
    Zx, Gamma, ZxT = np.linalg.svd(X, full_matrices=True, compute_uv=True, hermitian=True)
    pinverse = Ud@np.diag(Wd**-1)@Zx@(np.diag(np.diag(Ir+Gamma)**-1))@(Ud@np.diag(Wd**-1)@Zx).T
    # test
    return pinverse 

    
    
 


