# Demonstrates how to construct 2D variable coefficient second-derivative SBP operators, and verifies SBP properties.
import numpy as np
from orig.D2_Variable_orig import D2_Variable_4
import scipy.sparse as spsp

# Number of grid points
m = 31
N = m*m

# Space discretization
xvec,h = np.linspace(0,1,m,retstep=True)
yvec,h = np.linspace(0,1,m,retstep=True)
[X,Y] = np.meshgrid(xvec,yvec)
x = np.reshape(X.T,[N])
y = np.reshape(Y.T,[N])
           
# 1D SBP operators
H,HI,D1,D2_fun,e_l,e_r,d1_l,d1_r = D2_Variable_4(m,h)
Im = np.eye(m)

# Variable coefficient, some positive function of x and y
b = np.sqrt(x*y + 1)*np.exp(np.cos(x)**np.sin(y))

# 2D SBP operators
# Instead of Kronecker products
ind = np.reshape(np.array(range(N)),[m,m])
D2x = np.zeros((N,N))
D2y = np.zeros((N,N))
for i in range(0,m):
    D = D2_fun(b[ind[:,i]])
    p = ind[:,i]
    for count,idx in enumerate(p):
        D2x[p,idx] = D[:,count]
                        
for i in range(0,m):
    D = D2_fun(b[ind[i,:]])
    p = ind[i,:]
    for count,idx in enumerate(p):
        D2y[p,idx] = D[:,count]
          
HH = np.kron(H,H)
eW = np.kron(e_l,Im)
eE = np.kron(e_r,Im)
eS = np.kron(Im,e_l)
eN = np.kron(Im,e_r)
d1_W = np.kron(d1_l,Im)
d1_E = np.kron(d1_r,Im)
d1_S = np.kron(Im,d1_l)
d1_N = np.kron(Im,d1_r)

# Check SBP property. Eigenvalues should be real and non-positive.
Mx = -HH@D2x - np.diag(b)@eW.T@H@d1_W + np.diag(b)@eE.T@H@d1_E
My = -HH@D2y - np.diag(b)@eS.T@H@d1_S + np.diag(b)@eN.T@H@d1_N
       
[eigMx,_] = np.linalg.eig(Mx)
print(np.max(np.max(np.abs(Mx - Mx.T))))
print(np.max(np.abs(np.imag(eigMx))))
print(np.min(np.real(eigMx)))

[eigMy,_] = np.linalg.eig(My)
print(np.max(np.max(np.abs(My - My.T))))
print(np.max(np.abs(np.imag(eigMy))))
print(np.min(np.real(eigMy)))

