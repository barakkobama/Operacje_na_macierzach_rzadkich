import numpy as np
import scipy.linalg
import scipy as sp
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.cusolver as solver
from scipy.linalg import lu
import time

def LU_CPU(A):
    start = time.time()
    L,U= lu(A, True)
    end = time.time()
    return L, U, start-end

def LU_GPU(A):
    start = time.time()
    A = np.asarray(A).astype(np.float32)
    h = solver.cusolverDnCreate()
    m, n = A.shape
    A_GPU = gpuarray.to_gpu(A.T.copy())
    Lwork = solver.cusolverDnSgetrf_bufferSize(h, m, n, A_GPU.gpudata, m)
    workspace_gpu = gpuarray.zeros(Lwork, np.float32)
    devipiv_gpu = gpuarray.zeros(min(m, n), np.int32)
    devinfo_gpu = gpuarray.zeros(1, np.int32)
    solver.cusolverDnSgetrf(h, m, n, A_GPU.gpudata, m, workspace_gpu.gpudata, devipiv_gpu.gpudata, devinfo_gpu.gpudata)
    l_cuda = np.tril(A_GPU.get().T, -1)
    u_cuda = np.triu(A_GPU.get().T)
    if m < n:
        l_cuda = l_cuda[:, :m]
    else:
        u_cuda = u_cuda[:n, :]
    solver.cusolverDnDestroy(h)
    end = time.time()
    return l_cuda, u_cuda, start-end