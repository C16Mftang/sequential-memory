import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.utils import *
from src.models import *

result_path = os.path.join('./results/', 'asymmetric')
if not os.path.exists(result_path):
    os.makedirs(result_path)

temp_path = os.path.join('./results/', 'temporary_results')
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_correlated_binary_patterns(P, N, b, seed=1):
    np.random.seed(seed)
    X = np.zeros((int(P), int(N)))
    template = np.random.choice([-1, 1], size=N)
    prob = (1 + b) / 2
    for i in range(P):
        for j in range(N):
            if np.random.binomial(1, prob) == 1:
                X[i, j] = template[j]
            else:
                X[i, j] = -template[j]
            
            # revert the sign
            if np.random.binomial(1, 0.5) == 1:
                X[i, j] *= -1

    return to_torch(X, device)

learn_iters = 800
lr = 5e-1

def search_Pmax(Ns, b, ubound_P, search_step, model='1'):
    prev_P = 2
    
    # number of sweeps for each N and each P to reduce randomness
    K = 10 
    Pmaxs = []

    # initial search range of Ps
    Ps = np.arange(prev_P, ubound_P+search_step, search_step)

    for N in Ns:
        print('==========================')
        print(f'Current N:{N}')
        # set an initial value for Pmax so we can detect if the upper bound of P is exceeded
        Pmax = ubound_P

        losses_N = []
        # plt.figure(figsize=(3, 3))
        for P_ind, P in enumerate(Ps):
            n_errors = 0
            curr_losses = [] # K x learn_iters
            for k in range(K):
                # generate data, couple seed with k
                X = generate_correlated_binary_patterns(P, N, b, seed=k)
                X = X.to(torch.float)

                if model == 'PC':
                    # train PC
                    net = LinearSingleLayertPC(N, learn_iters, lr)
                    losses = net.train(X)
                    recall = net.recall(X[:-1])
                    curr_losses.append(losses)
                else:
                    # train HNs
                    net = ModernAsymmetricHopfieldNetwork(N, model)
                    net.train(X)
                    recall = net(X, X[:-1]) # shape (P-1)xN
                
                # groundtruth
                gt = X[1:]
                
                # compute the number of mismatches between recall and grountruth
                n_errors += torch.sum(recall != gt)
            
            # take the average of traning loss across K sweeps
            curr_losses = np.array(curr_losses)
            avg_losses_sweep = np.mean(curr_losses, axis=0)

            # a temporary block for tunning learning rates
            # plt.plot(avg_losses_sweep, label=f'P={P}')

            # compute the probability of errors as the percentage of mismatched bits across K sweeps
            error_prob = n_errors / ((P - 1) * N * K)
            print(f'Current P:{P}, error prob:{error_prob}')

            # once prob of error exceeds 0.01 we assign the previous P as Pmax
            if error_prob >= 0.01:
                Pmax = Ps[P_ind - 1]

                # stop increasing the value of P
                break

        print(f'Pmax:{Pmax}')
        
        """
        for next larger N, we are sure that its Pmax is larger than the curren one
        so we start the search from the current Pmax
        """
        Ps = np.arange(Pmax, ubound_P, search_step)

        # collect Pmax
        Pmaxs.append(Pmax)

        # save fig
        # plt.legend()
        # plt.savefig(temp_path + f'/losses_N={N}')
    
    return Pmaxs


Ns = np.arange(10, 110, 10)
b = 0

Pmaxs_pc = search_Pmax(Ns, b, search_step=1, ubound_P=100, model='PC')
Pmaxs_1 = search_Pmax(Ns, b, search_step=1, ubound_P=100, model='1')
Pmaxs_2 = search_Pmax(Ns, b, search_step=5, ubound_P=1000, model='2')
# Pmaxs_3 = search_Pmax(Ns, b, search_step=10, ubound_P=5000, sep='3')

plt.figure(figsize=(4, 3))
plt.plot(Ns, Ns, label='Identity', c='k', ls='--', marker='o')
plt.plot(Ns, Pmaxs_pc, label='PC', marker='o')
plt.plot(Ns, Pmaxs_1, label='HN (d=1)', marker='o', c='#13678A')
plt.plot(Ns, Pmaxs_2, label='HN (d=2)', marker='o', c='#45C4B0')
plt.yscale("log")
plt.legend()
plt.title('Capacity of models')
plt.xlabel(r'$N$')
plt.ylabel(r'$P_{max}$')
plt.xticks(Ns, Ns)
plt.tight_layout()
plt.savefig(result_path + f'/Capacity', dpi=200)



