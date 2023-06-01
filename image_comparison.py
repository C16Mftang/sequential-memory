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
parser.add_argument('--data-type', type=str, default='continuous', choices=['binary', 'continuous'],
                    help='for cifar/imagenet data this should always be continuous')
args = parser.parse_args()

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
            PC_losses = train_singlelayer_tPC(pc, optimizer, seq, learn_iters, device)
            # save the PC model for later recall - because training PC is exhausting
            torch.save(pc.state_dict(), PATH)
            _plot_PC_loss(PC_losses, seq_len, learn_iters)

        else:
            # recall mode, no training need, fast
            pc.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            pc.eval()

            with torch.no_grad():
                PC_recall = singlelayer_recall(pc, seq, device, args)
                HN_recall = hn_recall(hn, seq, device, args)

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
