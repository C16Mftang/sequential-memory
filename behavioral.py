import os
import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use('ggplot')
from src.models import ModernAsymmetricHopfieldNetwork, MultilayertPC, SingleLayertPC
from src.utils import *
from src.get_data import *

path = 'behavioral'
result_path = os.path.join('./results/', path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

num_path = os.path.join('./results/', path, 'numerical')
if not os.path.exists(num_path):
    os.makedirs(num_path)

fig_path = os.path.join('./results/', path, 'fig')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

model_path = os.path.join('./results/', path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

# add parser as varaible of the main class
parser = argparse.ArgumentParser(description='Sequential memories')

parser.add_argument('--query', type=str, default='offline', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
args = parser.parse_args()

def _plot_PC_loss(loss, seq_len):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}')
    plt.close()

def generate_binary_sequence(num_seqs, seq_len, k=0):
    """
    k: an auxilliary parameter to change seeds
    output: num_seqs x seq_len x d
    """

    # dimension of each element in the sequence.
    # equal to seq_len as it is a one-hot vector
    d = seq_len

    # the dataset
    seqs = np.zeros((num_seqs, seq_len, d))

    # shuffle the template to get unique new sequences
    for i in range(num_seqs):
        # the original, template sequence
        temp = np.arange(seq_len)
        np.random.seed(i + k)
        np.random.shuffle(temp) # seq_len

        # convert to one-hot vector
        one_hot_seq = np.eye(seq_len)[temp]
        seqs[i] = one_hot_seq
    
    return seqs

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# hyperparameters
num_seqs = 100
latent_size = 6
nonlin = 'tanh'
learn_lr = 5e-2
learn_iters = 300
inf_iters = 100
inf_lr = 1e-2
lens = range(7, 8)

accs = []
start_time = time.time()
for seq_len in lens:
    for k in [7000, 8000, 9000, 10000, 11000]:
        print(f'k={k}')
        seqs = generate_binary_sequence(num_seqs, seq_len, k=k)
        input_size = seq_len

        # number of perfectly recalled sequences under this seq_len
        n_perfect = 0
        # to collect all recalls; num_seqs x seq_len x d
        all_recalls = torch.zeros((num_seqs, seq_len))
        all_memories = torch.zeros((num_seqs, seq_len))

        # calculate the recalled item at each item position
        counts = np.zeros((seq_len, seq_len)) 
        for n in range(num_seqs):
            print(f'Length {seq_len}, Sequence {n}')
            seq = torch.from_numpy(seqs[n]).to(device, torch.float) # seq_len x d

            # initialize a new model for each sequence, each seq_len
            mpc = MultilayertPC(latent_size, input_size, nonlin=nonlin).to(device)
            m_optimizer = torch.optim.SGD(mpc.parameters(), lr=learn_lr)

            # train mPC
            print('Training multi layer tPC')
            mPC_losses = train_multilayer_tPC(mpc, m_optimizer, seq, learn_iters, inf_iters, inf_lr, device)
            _plot_PC_loss(mPC_losses, seq_len)

            # recall; seq_len x d
            with torch.no_grad():
                m_recalls = multilayer_recall(mpc, seq, inf_iters, inf_lr, args, device)
            
            # check if the target item is correctly recalled
            if torch.equal(seq.argmax(1), m_recalls.argmax(1)):
                n_perfect += 1

            # convert to list, length = seq_len
            mem_l = to_np(seq.argmax(1)).tolist()
            recall_l = to_np(m_recalls.argmax(1)).tolist()

            # at each position, check which position is recalled and count it
            # for example, if the recall at pos0 is what is at pos0 in memory,
            # counts[0,0]+1
            for i in range(seq_len):
                ind = mem_l.index(recall_l[i])
                counts[i, ind] += 1

        accs.append(n_perfect / num_seqs)
        print(counts)
        np.save(num_path + f'/counts{k}', counts)

# print(accs)
# np.save(num_path + '/seq_length_accs', np.array(accs))
print(f'Total time: {time.time() - start_time}')
