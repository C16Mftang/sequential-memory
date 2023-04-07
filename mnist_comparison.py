"""
This code compares the performance of tPC, AHN (with different separation function)
on a single MNIST sequence of 0-9
"""

import os
import argparse
import json
import time
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import *
from src.utils import *
from src.get_data import *

result_path = os.path.join('./results/', 'mnist_sequence')
if not os.path.exists(result_path):
    os.makedirs(result_path)

num_path = os.path.join('./results/', 'mnist_sequence', 'numerical')
if not os.path.exists(num_path):
    os.makedirs(num_path)

fig_path = os.path.join('./results/', 'mnist_sequence', 'fig')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# add parser as varaible of the main class
parser = argparse.ArgumentParser(description='Sequential memories')
parser.add_argument('--seq-len', type=int, default=10, 
                    help='length of input for training (default: 10)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for model init (default: 1)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='seed for model init (default: 1)')
parser.add_argument('--HN-type', type=str, default='softmax',
                    help='type of MAHN')
parser.add_argument('--data-type', type=str, default='continuous', choices=['binary', 'continuous'],
                    help='type of data; note that when HN type is exp or softmax, \
                        this should be always continuous')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--beta', type=int, default=1,
                    help='beta value for the MCHN')
args = parser.parse_args()


def train_PC(pc, optimizer, seq, learn_iters, device):
    seq_len = seq.shape[0]
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        prev = pc.init_hidden(1).to(device)
        batch_loss = 0
        for k in range(seq_len):
            x = seq[k]
            optimizer.zero_grad()
            energy = pc.get_energy(x, prev)
            energy.backward()
            optimizer.step()
            prev = x.clone().detach()

            # add up the loss value at each time step
            epoch_loss += energy.item() / seq_len
        losses.append(epoch_loss)

    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def _pc_recall(model, seq, query_type, binary):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N))
    recall[0] = seq[0].clone().detach()
    if query_type == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq[:-1])) if binary else model(seq[:-1])
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            recall[k] = torch.sign(model(recall[k-1])) if binary else model(recall[k-1]) # 1xN

    return recall
    
def _hn_recall(model, seq, query_type, binary):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N))
    recall[0] = seq[0].clone().detach()
    if query_type == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq, seq[:-1])) if binary else model(seq, seq[:-1]) # (P-1)xN
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            # prev = torch.sign(model(seq, prev)) if binary else model(seq, prev) # 1xN
            recall[k] = torch.sign(model(seq, recall[k-1])) if binary else model(seq, recall[k-1]) # 1xN

    return recall

def _plot_recalls(recall, model_name, args):
    seq_len = recall.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(recall[j].reshape(28, 28)), cmap='gray_r')
        ax[j].axis('off')
    plt.savefig(fig_path + f'/{model_name}_len{args.seq_len}_query{args.query}_data{args.data_type}', dpi=150)

def _plot_memory(x, seed):
    seq_len = x.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(x[j].reshape(28, 28)), cmap='gray_r')
        ax[j].axis('off')
    plt.savefig(fig_path + f'/memory_len{seq_len}_seed{seed}', dpi=150)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data
    w = 28

    # command line inputs
    seq_len = args.seq_len
    seed = args.seed
    learn_iters = args.epochs
    learn_lr = args.lr
    sep = args.HN_type
    beta = args.beta
    query_type = args.query
    binary = True if args.data_type == 'binary' else False

    print(f'Training variables: seq_len:{seq_len}')

    # load data
    seq = load_sequence_mnist(seed, seq_len, binary=binary).to(device)
    seq = seq.reshape((seq_len, w ** 2)) # seq_lenx784

    # temporal PC
    pc = SingleLayertPC(input_size=w ** 2, nonlin='linear').to(device)
    optimizer = torch.optim.Adam(pc.parameters(), lr=learn_lr)
    
    # HN with separation function
    hn = ModernAsymmetricHopfieldNetwork(w ** 2, sep=sep, beta=beta).to(device)

    # training PC
    # note that there is no need to train MAHN - we can just write down the retrieval
    PC_losses = train_PC(pc, optimizer, seq, learn_iters, device)
    
    with torch.no_grad():
        PC_recall = _pc_recall(pc, seq, query_type, binary)
    HN_recall = _hn_recall(hn, seq, query_type, binary)

    _plot_recalls(PC_recall, 'PC', args)
    HN_name = f'HN{sep}beta{beta}' if sep == 'softmax' else f'HN{sep}'
    _plot_recalls(HN_recall, HN_name, args)

    gt = seq[1:]
    # HN_mse = torch.mean((seq - HN_recall) ** 2)
    # PC_mse = torch.mean((seq - PC_recall) ** 2)
    HN_ssim = ssim(to_np(seq[-1]), to_np(HN_recall[-1]))
    PC_ssim = ssim(to_np(seq[-1]), to_np(PC_recall[-1]))

    # print((seq - HN_recall)[1])
    # print((seq - PC_recall)[1])

    print(PC_ssim, HN_ssim)
    # print(PC_mse, HN_mse)

    # plt.figure()
    # plt.plot(PC_losses, label='squared error sum')
    # plt.legend()
    # plt.savefig(result_path + f'/losses_len{seq_len}_iters{learn_iters}')

    _plot_memory(seq, seed)

if __name__ == "__main__":
    main(args)
