import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.utils import *
from src.models import *
from src.get_data import generate_correlated_binary_patterns

result_path = os.path.join('./results/', 'pvn')
if not os.path.exists(result_path):
    os.makedirs(result_path)

temp_path = os.path.join('./results/', 'temporary_results')
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='binary patterns')
parser.add_argument('--tol', type=float, default=0.01,
                    help='tolerance when determining Pmax')
parser.add_argument('--start-P', type=int, default=[2, 2, 2], nargs='+',
                    help='start of the number of patterns for search')
parser.add_argument('--ubound-P', type=int, default=[100, 100, 1000], nargs='+',
                    help='end of the number of patterns for search')
parser.add_argument('--search-step', type=int, default=[1, 1, 5], nargs='+',
                    help='search step of the number of patterns for search')
parser.add_argument('--models', type=str, default=['PC', '1', '2'], nargs='+',
                    help='model names')
args = parser.parse_args()


learn_iters = 800
lr = 5e-1

def search_Pmax(Ns, b, start_P, ubound_P, search_step, tol, model='1'):
    
    # number of sweeps for each N and each P to reduce randomness
    K = 10 
    Pmaxs = []

    """
    initial search range of Ps
    Note that the larger the P, the closer the mean of X (dim=0)
    is to 0 and the closer X^TX is to the real covariance
    """
    Ps = np.arange(start_P, ubound_P+search_step, search_step)

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
                X = generate_correlated_binary_patterns(P, N, b, device=device, seed=k)
                X = X.to(torch.float)

                if model == 'PC':
                    # train PC
                    net = LinearSingleLayertPC(N, learn_iters, lr)
                    losses = net.train(X)
                    recall = torch.sign(net.recall(X[:-1]))
                    curr_losses.append(losses)
                else:
                    # train HNs
                    net = ModernAsymmetricHopfieldNetwork(N, model)
                    net.train(X)
                    recall = torch.sign(net(X, X[:-1])) # shape (P-1)xN
                
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
            if error_prob >= tol:
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
        Pmaxs.append(int(Pmax))

        # save fig
        # plt.legend()
        # plt.savefig(temp_path + f'/losses_N={N}')
    
    return Pmaxs

def main(args):
    Ns = np.arange(10, 110, 10)
    b = 0

    results = {}
    for i, model in enumerate(args.models):
        Pmaxs = search_Pmax(Ns, 
                            b, 
                            start_P=args.start_P[i], 
                            search_step=args.search_step[i], 
                            ubound_P=args.ubound_P[i], 
                            tol = args.tol,
                            model=args.models[i])
        results[args.models[i]] = Pmaxs


    # Pmaxs_pc = search_Pmax(Ns, b, search_step=1, ubound_P=100, tol=tol, model='PC')
    # Pmaxs_1 = search_Pmax(Ns, b, search_step=1, ubound_P=100, tol=tol, model='1')
    # Pmaxs_2 = search_Pmax(Ns, b, search_step=5, ubound_P=1000, tol=tol, model='2')
    # Pmaxs_3 = search_Pmax(Ns, b, search_step=10, ubound_P=5000, sep='3')

    print(results)
    json.dump(results, open(result_path + f"/Pmaxs_tol{args.tol}.json", 'w'))

if __name__ == "__main__":
    main(args)



