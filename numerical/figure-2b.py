import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from olnm_np import SGD_Optimizer, OLNM_Optimizer, get_experiment_data

N = 10000       
d = 100         
num_steps = 1000
best_after= num_steps # for parameter tuning

L = 500
kappa = 500

np.random.seed(12345)
A_full, _ = get_experiment_data(N, d, L, kappa, seed=12345)
x_star = np.random.randn(d) * 0.1
b_full = A_full @ x_star + np.random.normal(0, 1e-3, N)
x_star = np.linalg.lstsq( A_full, b_full )[0] # optimization solution

def run_SGD(batchsize,learning_rate):
    sgd = SGD_Optimizer(d=d, m=batchsize, step_size=learning_rate)
    sgd_error_history = [np.linalg.norm(sgd.x - x_star)]
    for t in range(num_steps):
        # Sample a minibatch
        indices = np.random.choice(N, batchsize, replace=True)
        A_t = A_full[indices, :]
        b_t = b_full[indices]
        
        # Run SGD
        x_sgd = sgd.step(A_t, b_t)
        sgd_error = np.linalg.norm(x_sgd - x_star)
        sgd_error_history.append(sgd_error)
    return np.array(sgd_error_history)
def run_OLNM(batchsize,learning_rate,T):
    olnm = OLNM_Optimizer(d=d, L=1/learning_rate, kappa=kappa, m=batchsize, method='constant', c=T) 
    olnm_error_history = [np.linalg.norm(olnm.x - x_star)]
    for t in range(num_steps):
        # Sample a minibatch
        if olnm.reset is True:
            indices = np.random.choice(N, batchsize, replace=True)
            A_t = A_full[indices, :]
            b_t = b_full[indices]
            # Sample a new batch.  IMPORTANT!!!
            A_s = A_t
            b_s = b_t
        
        # Run OLNM
        x_olnm = olnm.step(A_s, b_s)
        olnm_error = np.linalg.norm(x_olnm - x_star)
        olnm_error_history.append(olnm_error)
    return np.array(olnm_error_history)

# --------
batchSizeList = np.array( [250] ) # 250 works well. Anything closer to d (100) is unpredictable. Anything larger, it's not worth it
stepSizeList  = np.logspace( -2,-4, 8)
T_List = [50,100,200]
# bestSoFar = 1e10

plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Helvetica'
for batchsize in batchSizeList:
    bestSoFar = 1e10
    for stepsize in stepSizeList:
        err = run_SGD( batchsize, stepsize )
        if np.isfinite(err).all():
            if err[best_after] < bestSoFar:
                # best_batchsize = batchsize
                best_stepsize  = stepsize
                best_err       = err
                bestSoFar      = err[best_after]
    plt.loglog( batchsize*np.arange(1,num_steps+2), best_err, '--',label=f'SGD,   b={batchsize:4d}, $\eta=$ {best_stepsize:.4f}' )
for batchsize in batchSizeList:
    for T in T_List:
        bestSoFar = 1e10
        for stepsize in stepSizeList:
            err = run_OLNM( batchsize, stepsize, T )
            if np.isfinite(err).all():
                if err[best_after] < bestSoFar:
                    # best_batchsize = batchsize
                    best_stepsize  = stepsize
                    best_err       = err
                    bestSoFar      = err[best_after]
        plt.loglog( batchsize*np.arange(1,num_steps+2), best_err, label=f'OLNM, b={batchsize:4d}, $\eta=$ {best_stepsize:.4f}, T={T:3d}' )    
plt.xlabel(r'Number of data sample calls ($b\times t$)', fontsize=16)
plt.ylabel('Distance to minimizer', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.savefig("imgs/png/figure-2bv1.png")
plt.savefig("imgs/pdf/figure-2bv1.pdf", bbox_inches="tight")
plt.savefig("imgs/svg/figure-2bv1.svg", bbox_inches="tight")
# plt.show()


# ----------
batchSizeList = np.array( [250,500] ) # 250 works well. Anything closer to d (100) is unpredictable. Anything larger, it's not worth it
stepSizeList  = [.001] # don't tune it
T_List = [50,100,200]

plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Helvetica'
for batchsize in batchSizeList:
    bestSoFar = 1e10
    for stepsize in stepSizeList:
        err = run_SGD( batchsize, stepsize )
        if np.isfinite(err).all():
            if err[best_after] < bestSoFar:
                # best_batchsize = batchsize
                best_stepsize  = stepsize
                best_err       = err
                bestSoFar      = err[best_after]
    plt.loglog( batchsize*np.arange(1,num_steps+2), best_err, '--',label=f'SGD,   b={batchsize:4d}, $\eta=$ {best_stepsize:.4f}' )
for batchsize in batchSizeList:
    for T in T_List:
        bestSoFar = 1e10
        for stepsize in stepSizeList:
            err = run_OLNM( batchsize, stepsize, T )
            if np.isfinite(err).all():
                if err[best_after] < bestSoFar:
                    # best_batchsize = batchsize
                    best_stepsize  = stepsize
                    best_err       = err
                    bestSoFar      = err[best_after]
        plt.loglog( batchsize*np.arange(1,num_steps+2), best_err, label=f'OLNM, b={batchsize:4d}, $\eta=$ {best_stepsize:.4f}, T={T:3d}' )    
plt.xlabel(r'Number of data sample calls ($b\times t$)', fontsize=16)
plt.ylabel('Distance to minimizer', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.savefig("imgs/png/figure-2bv2.png")
plt.savefig("imgs/pdf/figure-2bv2.pdf", bbox_inches="tight")
plt.savefig("imgs/svg/figure-2bv2.svg", bbox_inches="tight")
# plt.show()