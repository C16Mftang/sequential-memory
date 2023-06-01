"""A simple example of how to train a 2-layer tPC without any comparison to HNs"""

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
from src.models import MultilayertPC
from src.utils import *
from src.get_data import *

path = 'multilayer'
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
parser.add_argument('--seq-len-max', type=int, default=11, 
                    help='max power of length of input for training (default: 11),\
                          specify multiple values separated by whitespace')
parser.add_argument('--seed', type=int, default=[1], nargs='+',
                    help='seed for model init (default: 1); can be multiple, separated by space')
parser.add_argument('--latent-size', type=int, default=480,
                    help='hidden size for training 480 for mnist; 1900 for cifar10')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--nonlinearity', type=str, default='tanh',
                    help='nonlinear function used in the model')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'recall'],
                    help='mode of the script: train or recall (just to save time)')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar'],
                    help='which dataset to memorize')
parser.add_argument('--repeat', type=float, default=0,
                    help='percentage of repeating digits')
args = parser.parse_args()

def _plot_recalls(recall, args):
    seq_len = recall.shape[0]

    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        if args.data == 'mnist':
            img = to_np(recall[j].reshape((28, 28))) 
        else:
            img = to_np(recall[j].reshape((3, 32, 32)).permute(1, 2, 0))

        ax[j].imshow(img, cmap='gray_r')
        ax[j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/mtPC_len{seq_len}_query{args.query}_{args.data}_repeat{int(args.repeat*100)}percen', dpi=150)

def _plot_memory(x, args):
    seq_len = x.shape[0]

    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        if args.data == 'mnist':
            img = to_np(x[j].reshape((28, 28))) 
        else:
            img = to_np(x[j].reshape((3, 32, 32)).permute(1, 2, 0))

        ax[j].imshow(img, cmap='gray_r')
        ax[j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/memory_len{seq_len}_{args.data}_repeat{int(args.repeat*100)}percen', dpi=150)

def _plot_PC_loss(loss, seq_len, learn_iters, dataset):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}_iters{learn_iters}_{dataset}')
        
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data and model
    seq_len_max = args.seq_len_max
    learn_iters = args.epochs
    learn_lr = args.lr
    latent_size = args.latent_size
    seed = args.seed
    mode = args.mode
    dataset = args.data
    input_size = 784 if dataset == 'mnist' else 3072
    
    # inference variables: no need to tune too much
    inf_iters = 100
    inf_lr = 1e-2

    MSEs = []
    seq_lens = [2 ** pow for pow in range(1, seq_len_max)]
    for seq_len in seq_lens:
        
        # varying lr for different datasets
        if dataset == 'cifar':
            if seq_len == 16:
                learn_lr /= 2
            if seq_len == 32:
                learn_lr /= 2
            if seq_len == 128:
                learn_lr /= 2
            if seq_len == 512:
                learn_lr /= 2
                
        elif dataset == 'mnist':
            if seq_len == 64:
                learn_lr /= 2
            if seq_len == 256:
                learn_lr /= 2
            if seq_len == 512:
                learn_lr /= 2

        print(f'Training variables: seq_len:{seq_len}; seed:{seed}; lr:{learn_lr}')

        # load data
        if dataset == 'mnist':
            seq = load_sequence_mnist(seed, seq_len, order=False, binary=False).to(device)
        elif dataset == 'cifar':
            seq = load_sequence_cifar(seed, seq_len).to(device)
        # ...or any other custom dataset

        # if we want to have repeating digits
        if args.repeat > 0:
            seq = replace_images(seq, seed=seed, p=args.repeat)
        seq = seq.reshape((seq_len, input_size))
        
        # multilayer PC
        model = MultilayertPC(latent_size, input_size, nonlin='tanh').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

        # path to save model to/load model from
        PATH = os.path.join(model_path, f'PC_len{seq_len}_seed{seed}_{dataset}_repeat{int(args.repeat*100)}percen.pt')

        if mode == 'train':
            # train PC
            PC_losses = train_multilayer_tPC(model, optimizer, seq, learn_iters, inf_iters, inf_lr, device)
            # save the current model and plot the loss for tunning
            torch.save(model.state_dict(), PATH)
            _plot_PC_loss(PC_losses, seq_len, learn_iters, dataset)
        
        elif mode == 'recall':
            # recall mode, no training need, fast
            model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            model.eval()

            # slighlt increase the inferenc iters during retrieval
            inf_iters = 200

            with torch.no_grad():
                recalls = multilayer_recall(model, seq, inf_iters, inf_lr, args, device)

            if seq_len <= 16:
                _plot_recalls(recalls, args)
                _plot_memory(seq, args)
            
            MSEs.append(float(to_np(torch.mean((seq - recalls) ** 2))))

    if mode == 'recall':
        results = {}
        results["PC"] = MSEs
        json.dump(results, open(num_path + f"/MSEs_seed{seed}_query{args.query}_{args.data}_repeat{int(args.repeat*100)}percen.json", 'w'))

if __name__ == "__main__":
    for s in args.seed:
        start_time = time.time()
        args.seed = s
        main(args)
        print(f'Seed complete, total time: {time.time() - start_time}')
