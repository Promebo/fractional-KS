#%%
import scipy
import math
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy.optimize as opt

# parameter one hole
sites = 50 # need even site
t0 = 1
j0 = 0.3 # t-J model parameter
N = 100 # iteration number
sigma = 0.01 # smearing Gaussina width
mix = 0.4  # mixing parameter
holes = 1 # number of hole

# LDA parameters for hopping parameter
A = 0.45
B = 0.12

# external potential
v_ext = np.zeros(sites)

# initail guessing for open boundary
n = np.ones(sites)*(sites-1)/sites # guess the initial density
lanmbda= np.ones(sites)*0.1 # initial chemical potential
t = np.ones(sites-1)*0.6 # initial t_h
j = np.ones(sites-1)*0.3 # initial t_f

# save to check convergence of t_h, t_f, and chemical potential
T = [t[0]]
J = [j[0]]
X = [lanmbda[5]]
F = [0]

# exchange correlation
def v_xc_h(ni):
    v_I = -0.2136899*ni+0.12327306-ni*0.2136899
    v_T = A*4/np.pi + 0.3*B*(ni-1) - A*B*3*np.pi*(ni-1)*(ni-1)/2 - 0.3*np.pi*np.pi*B*B*(ni-1)*(ni-1)*(ni-1)/8
    vi = v_I + v_T
    return vi

# find gaussian chemical potential
def function_miu(miu):
    gaussian_f = np.zeros(sites)
    for ii in range(len(gaussian_f)):
        gaussian_f[ii] = 0.5*(1-math.erf((E_s[ii]-miu)/sigma))
    density_s = np.zeros(sites)

    for i in range(sites):
        density_s += fai_s[:,i]*fai_s[:,i]*gaussian_f[i]
    return (np.sum(density_s)-(sites-holes)/2)

# find chemical potential to fufill no-double-occupied constrain
def functionH(lanmbda):
    lanmbda = lanmbda - np.mean(lanmbda)
    aa=(lanmbda-v)
    H_diag = np.diag(aa)
    H_2 = np.diag(-t)
    a = np.zeros((sites-1))
    b = np.zeros(sites)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up =  np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_h = H_diag + H_up + H_dw
    H_h[0,sites-1] = 0
    H_h[sites-1,0] = 0  # open boundary
    E_h, fai_h = eigh(H_h)

    bb=(lanmbda)
    H_diag = np.diag(bb)
    H_2 = np.diag(-j)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up =  np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_s = H_diag + H_up + H_dw
    H_s[0,sites-1] = 0
    H_s[sites-1,0] = 0  # open boundary
    global E_s 
    global fai_s
    E_s, fai_s = eigh(H_s)

    density_h = 0
    for i in range(holes):
        density_h += fai_h[:,i]*fai_h[:,i]

    density_s = np.zeros(sites)

    miu = E_s[sites//2-holes//2-1] #guess
    optimizeresut = opt.root(function_miu, miu,tol=0.0001)
    miu = optimizeresut.x
    gaussian_f = np.zeros(len(E_s))
    for ii in range(len(gaussian_f)):
        gaussian_f[ii] = 0.5*(1-math.erf((E_s[ii]-miu)/sigma))

    for i in range(sites):
        density_s += fai_s[:,i]*fai_s[:,i]*gaussian_f[i]*2
    f1= density_h + density_s - 1

    return f1



for ii in range(N):
    # v = v_ext # w/o xc
    v = (v_ext+ v_xc_h(n))
    initial_guess = lanmbda
    optimizeresut = scipy.optimize.root(functionH, initial_guess,method="hybr",tol=0.0001)
    lanmbda = optimizeresut.x
    lanmbda0 = lanmbda - np.mean(lanmbda)

    #lanmbda0 = np.zeros(len(lanmbda0)) # w/o lambda

    aa=(lanmbda0-v)
    H_diag = np.diag(aa)
    H_2 = np.diag(-t)
    a = np.zeros((sites-1))
    b = np.zeros(sites)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up =  np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_h = H_diag + H_up + H_dw
    H_h[0,sites-1] = 0
    H_h[sites-1,0] = 0  # open boundary
    E_h, fai_h = eigh(H_h)

    bb=(lanmbda0)
    H_diag = np.diag(bb)
    H_2 = np.diag(-j)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up =  np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_s = H_diag + H_up + H_dw
    H_s[0,sites-1] = 0
    H_s[sites-1,0] = 0  # open boundary
    E_s, fai_s = eigh(H_s)

    if sites%2 != 0 :
        print("num of sites is not even!!")
        break

    density_h = 0
    for i in range(holes):
        density_h += fai_h[:,i]*fai_h[:,i]

    density_s = np.zeros(sites)
    miu = E_s[sites//2-holes//2-1] #guess
    optimizeresut = opt.root(function_miu, miu,tol=0.0001)
    miu = optimizeresut.x
    gaussian_f = np.zeros(len(E_s))
    for iii in range(len(gaussian_f)):
        gaussian_f[iii] = 0.5*(1-math.erf((E_s[iii]-miu)/sigma))
    
    for i in range(sites):
        density_s += fai_s[:,i]*fai_s[:,i]*gaussian_f[i]*2
    
    hihj = np.zeros(sites-1)
    for jj in range(holes):
        for i in range(0,sites-1):
            hihj[i] += fai_h[i,jj]*fai_h[i+1,jj]

    bibj = np.zeros(sites-1)
    for i in range(0,sites-1):
        if sites%2 == 0:
            #for one hole, even sites
            for jj in range(0,sites):
                bibj[i] += fai_s[i,jj]*fai_s[i+1,jj]*2*gaussian_f[jj]


    f0 = density_h + density_s - 1

    n = density_s
    #n = 1 - density_h
    n_right = np.zeros(len(n))
    for iii in range(0,len(n)-1):
        n_right[iii] = n[iii+1]
    n_right[-1] = n[0]
    n_mean = (n[:-1]+n_right[:-1])/2

    t_n = 2/np.pi - B*(n_mean-1)*(n_mean-1)*np.pi/4
    j_n = A*(-n_mean + 1) + 0.3/np.pi - B*(n_mean-1)*(n_mean-1)*0.3*np.pi/8

    ## LDA
    t_new = t_n 
    j_new = j_n


    t_next = mix*t + (1-mix)*(t_new)
    j_next = mix*j + (1-mix)*(j_new)

    t = t_next
    j = j_next
    T.append(t[0])
    J.append(j[0])
    X.append(lanmbda0[5])
    F.append(sum(abs(f0)))

    if sum(abs(f0)) < 0.00001 :
        print(ii,end="  ")

density = 1 - density_h
density = density_s
E0 = 0
E0 += -2*np.dot(t,hihj)
E0 += -2*np.dot(j,bibj)
E0 += np.dot(v_ext,density)
E0 += np.dot(density,-0.2136899*density+0.12327306) # adding xc
#output
print()
print("sum of f0 is :", sum(abs(f0)))
print("num of elec. :",sum(density))
print("num of holes :",holes)
print("density is :",density)
plt.plot(density)

print()
print("energy is :", E0)
print("kinetic is:",-2*np.dot(t,hihj)-2*np.dot(j,bibj))
print("external energy is:",np.dot(v_ext,density))
print("xc energy is",np.dot(density,-0.2136899*density+0.12327306))

# %%
