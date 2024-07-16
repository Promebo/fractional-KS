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
sigma = 0.01 # smearing Gaussina width
N = 100 # iteration number
mix = 0.4 # mixing parameter
#vvv = 0.0
holes = 1

# LDA parameters for hopping parameter
A = 0.45
B = 0.12

## external potential
stagger_num = sites # sites/2 use for stagger potential, need to be the factor of sites
v_stagger = np.zeros(stagger_num)
v_stagger[stagger_num//2] = -1 # one impurity
v_ext = np.zeros(sites)
for i in range(0,sites):
    for jjj in range(0,stagger_num):
        if i%stagger_num == jjj:
            v_ext[i] = v_stagger[jjj]

# #cos external
# periodicN = 50 
# print('periodicN is:',periodicN)
# g1 = 1
# for i in range(0,sites):
#     v_ext[i] = g1*np.cos(2*np.pi*i/periodicN)

# initail guess
n = np.ones(sites)*(sites-1)/sites # guess the initial density
lanmbda_one = np.ones(sites)*0.1  # initial chemical potential
t = np.ones(sites)*(0.63) # initial t_h
j = np.ones(sites)*(0.17) # initial t_f

# save to check convergence of t_h, t_f, and chemical potential
T = [t[0]]
J = [j[0]]
X = [lanmbda_one[0]]
Rest = []
Dens = [(50-1)/50]

# exchange correlation potential
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
def functionH(lanmbda_one):
    lanmbda_one = lanmbda_one - np.mean(lanmbda_one)
    lanmbda = np.zeros(sites)
    for i in range(0,sites):
        for jj in range(0,stagger_num):
            if i%stagger_num == jj:
                lanmbda[i] = lanmbda_one[jj]
    diag_term_h= (lanmbda-v)

    H_diag = np.diag(diag_term_h)
    H_2 = np.diag(-t[:-1])
    a = np.zeros((sites-1))
    b = np.zeros(sites)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up = np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_h = H_diag + H_up + H_dw
    if holes%2 != 0:
        H_h[0,sites-1] = -t[-1]
        H_h[sites-1,0] = -t[-1] # periodic boundary
    if holes%2 == 0:
        H_h[0,sites-1] = t[-1]
        H_h[sites-1,0] = t[-1] # antiperiodic boundary
    E_h, fai_h = eigh(H_h)

    diag_term_b= (lanmbda)
    H_diag = np.diag(diag_term_b)
    H_2 = np.diag(-j[:-1])
    a = np.zeros((sites-1))
    b = np.zeros(sites)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up = np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_s = H_diag + H_up + H_dw
    H_s[0,sites-1] = -j[-1]
    H_s[sites-1,0] = -j[-1] # periodic boundary
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
    return f1[0:stagger_num]


for ii in range(N):
    #v = v_ext  # w/o xc
    v = (v_ext + v_xc_h(n)) 
    initial_guess = lanmbda_one[0:stagger_num]
    optimizeresut = opt.root(functionH, initial_guess,tol=0.0001)
    lanmbda_one = optimizeresut.x
    lanmbda_one = lanmbda_one - np.mean(lanmbda_one)
    lanmbda0 = np.zeros(sites)
    for i in range(0,sites):
        for jj in range(0,stagger_num):
            if i%stagger_num == jj:
                lanmbda0[i] = lanmbda_one[jj]
    
    #lanmbda0 = np.zeros(len(lanmbda0)) # w/o lambda

    diag_term_h= (lanmbda0-v)
    H_diag = np.diag(diag_term_h)
    H_2 = np.diag(-t[:-1])
    a = np.zeros((sites-1))
    b = np.zeros(sites)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up = np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_h = H_diag + H_up + H_dw
    if holes%2 != 0:
        H_h[0,sites-1] = -t[-1]
        H_h[sites-1,0] = -t[-1] # periodic boundary
    if holes%2 == 0:
        H_h[0,sites-1] = t[-1]
        H_h[sites-1,0] = t[-1] # antiperiodic boundary
    E_h, fai_h = eigh(H_h)

    diag_term_b= (lanmbda0)
    H_diag = np.diag(diag_term_b)
    H_2 = np.diag(-j[:-1])
    a = np.zeros((sites-1))
    b = np.zeros(sites)
    H_up = np.insert(H_2,0,values=a,axis=0)
    H_up = np.insert(H_up,sites-1,values=b,axis=1)
    H_dw = np.insert(H_2,(sites-1),values=a,axis=0)
    H_dw = np.insert(H_dw,0,values=b,axis=1)
    H_s = H_diag + H_up + H_dw
    H_s[0,sites-1] = -j[-1]
    H_s[sites-1,0] = -j[-1] # periodic boundary
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

    f0= density_h + density_s - 1

    if np.sum(abs(f0)) < 0.0001 :
        print(ii,end=" ")
        #break

    hihj = np.zeros(sites)
    for jj in range(holes):
        for i in range(0,sites-1):
            hihj[i] += fai_h[i,jj]*fai_h[i+1,jj]
        hihj[sites-1] += fai_h[sites-1,jj]*fai_h[0,jj]
    
    
    bibj = np.zeros(sites)
    for i in range(0,sites-1):
        if sites%2 == 0:
            #for one hole, even sites
            for jj in range(0,sites):
                bibj[i] += fai_s[i,jj]*fai_s[i+1,jj]*2*gaussian_f[jj]

    if sites%2 == 0:
        for jj in range(0,sites):
            bibj[sites-1] += fai_s[sites-1,jj]*fai_s[0,jj]*2*gaussian_f[jj]

    n = 1-density_h
    n = density_s

    n_right = np.zeros(len(n))
    for iii in range(0,len(n)-1):
        n_right[iii] = n[iii+1]
    n_right[-1] = n[0]
    n_mean = (n+n_right)/2
    
    t_n = 2/np.pi - B*(n_mean-1)*(n_mean-1)*np.pi/4
    j_n = A*(-n_mean + 1) + 0.3/np.pi - B*(n_mean-1)*(n_mean-1)*0.3*np.pi/8

    ## mean -field
    t_new = t0*bibj
    j_new = 0.5*j0*bibj+t0*hihj
    ## LDA
    t_new = t_n 
    j_new = j_n


    t_next = mix*t + (1-mix)*(t_new)
    j_next = mix*j + (1-mix)*(j_new)
    Rest.append(np.sum(np.sum(t_next-t)))

    t = t_next
    j = j_next

    T.append(t[0])
    J.append(j[0])
    X.append(lanmbda_one[0])

    Dens.append(n[0])

density = 1-density_h
#density = density_s
E0 = 0
E0 += -2*np.dot(t,hihj)
E0 += -2*np.dot(j,bibj)
E0 += np.dot(v_ext,density)
#E0 += 2*t0*np.dot(hihj,bibj)+0.5*j0*np.dot(bibj,bibj) # MF
E0 += np.dot(density,-0.2136899*density+0.12327306) # adding xc


#output

print()
print("sum of f0 is :", np.sum(abs(f0)))
print("num of elec. :",np.sum(density))
print("num of hole:",holes)
print("density is :",density)
for i in range(len(density)):
    print(density[i],end=' ')
print()
print("v is:",v_ext)
print("energy is :", E0)
print("kinetic is:",-2*np.dot(t,hihj)-2*np.dot(j,bibj))
print("external energy is:",np.dot(v_ext,density))
print("xc energy is",np.dot(density,-0.2136899*density+0.12327306))
print()
plt.plot(density)


# %%
