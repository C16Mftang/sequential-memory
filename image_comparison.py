"""
This code compares the performance of tPC, AHN (with different separation function)
on a random MNIST sequence of length seq_len
"""

import os
import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import *
from src.utils import *
from src.get_data import *

cifar_path = 'cifar_sequence'
result_path = os.path.join('./results/', cifar_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

num_path = os.path.join('./results/', cifar_path, 'numerical')
if not os.path.exists(num_path):
    os.makedirs(num_path)

fig_path = os.path.join('./results/', cifar_path, 'fig')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

model_path = os.path.join('./results/', cifar_path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

# add parser as varaible of the main class
parser = argparse.ArgumentParser(description='Sequential memories')
parser.add_argument('--seq-len-max', type=int, default=11, 
                    help='max input length')
parser.add_argument('--seq-len-step', type=int, default=1, 
                    help='seq len increase rate')
parser.add_argument('--seed', type=int, default=[1], nargs='+',
                    help='seed for model init (default: 1); can be multiple, separated by space')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--nonlinearity', type=str, default='tanh',
                    help='nonlinear function used in the model')
parser.add_argument('--HN-type', type=str, default='softmax',
                    help='type of MAHN default to softmax')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'recall'],
                    help='mode of the script: train or recall (just to save time)')
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
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')

    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def _pc_recall(model, seq, query_type, device, binary=False):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    if query_type == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq[:-1])) if binary else model(seq[:-1])
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            recall[k] = torch.sign(model(recall[k-1:k])) if binary else model(recall[k-1:k]) # 1xN

    return recall
    
def _hn_recall(model, seq, query_type, device, binary=False):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    if query_type == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq, seq[:-1])) if binary else model(seq, seq[:-1]) # (P-1)xN
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            # prev = torch.sign(model(seq, prev)) if binary else model(seq, prev) # 1xN
            recall[k] = torch.sign(model(seq, recall[k-1:k])) if binary else model(seq, recall[k-1:k]) # 1xN

    return recall

def _plot_recalls(recall, model_name, args):
    seq_len = recall.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(recall[j].reshape((3, 32, 32)).permute(1, 2, 0)))
        ax[j].axis('off')
        ax[j].set_aspect("auto")
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fig_path + f'/{model_name}_len{seq_len}_query{args.query}', bbox_inches='tight', dpi=200)

def _plot_memory(x, seed):
    seq_len = x.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(x[j].reshape((3, 32, 32)).permute(1, 2, 0)))
        ax[j].axis('off')
        ax[j].set_aspect("auto")
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fig_path + f'/memory_len{seq_len}_seed{seed}', bbox_inches='tight', dpi=200)

def _plot_PC_loss(loss, seq_len, learn_iters):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}_iters{learn_iters}')

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data
    w = 32
    input_size = 3 * (w ** 2)

    # command line inputs
    seed = args.seed
    learn_iters = args.epochs
    learn_lr = args.lr
    sep = args.HN_type
    seq_len_max = args.seq_len_max
    seq_len_step = args.seq_len_step
    query_type = args.query
    mode = args.mode
    beta = args.beta
    nonlin = args.nonlinearity

    # loop through different seq_len
    PC_MSEs, HN_MSEs = [], []

    seq_lens = [2 ** pow for pow in range(1, seq_len_max)]
    # seq_lens = [10]
    for seq_len in seq_lens:

        if seq_len == 128:
            learn_iters += 200
        if seq_len == 256:
            learn_iters += 200
        if seq_len == 512:
            learn_iters += 200
            learn_lr += 1e-5
        if seq_len == 1024:
            learn_iters += 200
            learn_lr += 1e-5

        print(f'Training variables: seq_len:{seq_len}; seed:{seed}; lr:{learn_lr}; epoch:{learn_iters}')

        # load data
        seq = load_sequence_cifar(seed, seq_len).to(device)
        seq = seq.reshape((seq_len, input_size)) # seq_lenx3072

        # temporal PC
        pc = SingleLayertPC(input_size=input_size, nonlin=nonlin).to(device)
        optimizer = torch.optim.Adam(pc.parameters(), lr=learn_lr)

        # HN with linear separation function
        hn = ModernAsymmetricHopfieldNetwork(input_size, sep=sep, beta=beta).to(device)

        if nonlin != 'linear':
            PATH = os.path.join(model_path, f'PC_{nonlin}_len{seq_len}_seed{seed}.pt') 
        else:
            PATH = os.path.join(model_path, f'PC_len{seq_len}_seed{seed}.pt') 
        if mode == 'train':
            # training PC
            # note that there is no need to train MAHN - we can just write down the retrieval
            PC_losses = train_PC(pc, optimizer, seq, learn_iters, device)
            # save the PC model for later recall - because training PC is exhausting
            torch.save(pc.state_dict(), PATH)
            _plot_PC_loss(PC_losses, seq_len, learn_iters)

        else:
            # recall mode, no training need, fast
            pc.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            pc.eval()

            with torch.no_grad():
                PC_recall = _pc_recall(pc, seq, query_type, device)
                HN_recall = _hn_recall(hn, seq, query_type, device)

            if seq_len <= 32:
                PC_name = f'PC_{nonlin}' if nonlin != 'linear' else 'PC'
                _plot_recalls(PC_recall, PC_name, args)
                HN_name = f'HN{sep}beta{beta}' if sep == 'softmax' else f'HN{sep}'
                _plot_recalls(HN_recall, HN_name, args)

                # plot the original memories
                _plot_memory(seq, seed)

            # calculate MSE at each one, save file name with seed
            PC_MSEs.append(float(to_np(torch.mean((seq - PC_recall) ** 2))))
            HN_MSEs.append(float(to_np(torch.mean((seq - HN_recall) ** 2))))

    # save everything at this particular seed
    if mode == 'recall':
        results = {}
        results["PC"] = PC_MSEs
        results["HN"] = HN_MSEs
        print(results)
        json.dump(results, open(num_path + f"/MSEs_seed{seed}_query{query_type}_{nonlin}.json", 'w'))


if __name__ == "__main__":
    for s in args.seed:
        start_time = time.time()
        args.seed = s
        main(args)
        print(f'Seed complete, total time: {time.time() - start_time}')
