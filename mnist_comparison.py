"""
This code compares the performance of tPC, AHN (with different separation function)
on a random MNIST sequence of length seq_len
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

path = 'mnist_sequence'
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
                    help='max input length, power of 2')
parser.add_argument('--seed', type=int, default=[1], nargs='+',
                    help='seed for model init (default: 1); can be multiple, separated by space')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--nonlinearity', type=str, default='linear',
                    help='nonlinear function used in the model')
parser.add_argument('--HN-type', type=str, default='softmax',
                    help='type of MAHN default to softmax')
parser.add_argument('--data-type', type=str, default='continuous', choices=['binary', 'continuous'],
                    help='type of data; note that when HN type is exp or softmax, \
                        this should be always continuous')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'recall'],
                    help='mode of the script: train or recall (just to save time)')
parser.add_argument('--order', type=str, default='unorder', choices=['order', 'unorder'],
                    help='whether to load digits following 0-9 order')
parser.add_argument('--beta', type=int, default=1,
                    help='beta value for the MCHN')
parser.add_argument('--repeat', type=float, default=0,
                    help='percentage of repeating digits')
parser.add_argument('--noise', type=float, default=0,
                    help='std of noise added to the query')
args = parser.parse_args()

# local plotting functions for exploratory purposes
def _plot_recalls(recall, model_name, args):
    seq_len = recall.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(recall[j].reshape(28, 28)), cmap='gray_r')
        ax[j].axis('off')
    plt.tight_layout()
    # plt.savefig(fig_path + f'/{model_name}_len{seq_len}_query{args.query}_data{args.data_type}_repeat{int(args.repeat*100)}percen', dpi=150)
    plt.savefig(fig_path + f'/{model_name}_len{seq_len}_query{args.query}_data{args.data_type}_noise{int(args.noise*100)}', dpi=150)

def _plot_memory(x, seed, args):
    seq_len = x.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(x[j].reshape(28, 28)), cmap='gray_r')
        ax[j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/memory_len{seq_len}_seed{seed}_data{args.data_type}_repeat{int(args.repeat*100)}percen', dpi=150)

def _plot_PC_loss(loss, seq_len, learn_iters, data_type='continuous'):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}_iters{learn_iters}_{data_type}')

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data
    w = 28
    input_size = 1 * (w ** 2)

    # command line inputs
    seed = args.seed
    learn_iters = args.epochs
    learn_lr = args.lr
    sep = args.HN_type
    seq_len_max = args.seq_len_max
    query_type = args.query
    mode = args.mode
    beta = args.beta
    binary = True if args.data_type == 'binary' else False
    order = True if args.order == 'order' else False
    nonlin = args.nonlinearity

    # loop through different seq_len
    PC_MSEs = []
    HN_MSEs = []

    # seq_lens = [2 ** pow for pow in range(1, seq_len_max)] if not order else [2, 3, 5, 10]
    seq_lens = [8]
    for seq_len in seq_lens:
        if seq_len == 128:
            learn_iters += 100
        if seq_len == 256:
            learn_iters += 200
        if seq_len == 512:
            learn_iters += 200
        if seq_len == 1024:
            learn_iters += 200

        print(f'Training variables: seq_len:{seq_len}; seed:{seed}')

        # load data
        # make sure the percentage of repeating data points is indeed a percentage
        assert(args.repeat < 1)
        seq = load_sequence_mnist(seed, seq_len, order=order, binary=binary).to(device)

        # if we want to have repeating digits for aliased examples
        if args.repeat > 0:
            seq = replace_images(seq, seed=seed, p=args.repeat)

        # flatten the inputs
        seq = seq.reshape((seq_len, input_size)) # seq_lenx784

        # temporal PC
        pc = SingleLayertPC(input_size=input_size, nonlin=nonlin).to(device)
        optimizer = torch.optim.Adam(pc.parameters(), lr=learn_lr)

        # HN with linear separation function
        hn = ModernAsymmetricHopfieldNetwork(input_size, sep=sep, beta=beta).to(device)

        # PATH = os.path.join(model_path, f'PC_len{seq_len}_seed{seed}_{args.data_type}_repeat{int(args.repeat*100)}percen.pt')
        PATH = os.path.join(model_path, f'PC_len{seq_len}_seed{seed}_{args.data_type}.pt')
        if mode == 'train':
            # training PC
            # note that there is no need to train MAHN - we can just write down the retrieval
            PC_losses = train_singlelayer_tPC(pc, optimizer, seq, learn_iters, device)
            # save the PC model for later recall - because training PC is exhausting
            torch.save(pc.state_dict(), PATH)
            _plot_PC_loss(PC_losses, seq_len, learn_iters, data_type=args.data_type)

        else:
            # recall mode, no training need, fast
            pc.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            pc.eval()

            with torch.no_grad():
                PC_recall = singlelayer_recall(pc, seq, device, args)
                HN_recall = hn_recall(hn, seq, device, args)

            # online visualize the recalls when seq_len is small
            if seq_len <= 16:
                _plot_recalls(PC_recall, 'PC', args)
                HN_name = f'HN{sep}beta{beta}' if sep == 'softmax' else f'HN{sep}'
                _plot_recalls(HN_recall, HN_name, args)

                # plot the original memories
                _plot_memory(seq, seed, args)

            # calculate MSE at each one, save file name with seed
            PC_MSEs.append(float(to_np(torch.mean((seq - PC_recall) ** 2))))
            HN_MSEs.append(float(to_np(torch.mean((seq - HN_recall) ** 2))))

            # calculate SSIM
            """
            However this is not a really good measurement for memory retrievals

            Because HN will always retrieve some digit for us, whose ssim maybe similar to the correct
            memory but which can be totally wrong
            """
            
            # PC_SSIM, HN_SSIM = 0, 0
            # for k in range(seq_len):
            #     PC_SSIM += float(ssim(to_np(seq[k]), to_np(PC_recall[k]))) / seq_len
            #     HN_SSIM += float(ssim(to_np(seq[k]), to_np(HN_recall[k]))) / seq_len
            # PC_SSIMs.append(PC_SSIM)
            # HN_SSIMs.append(HN_SSIM)

    # save everything at this particular seed
    if mode == 'recall':
        results = {}
        results["PC"] = PC_MSEs
        results["HN"] = HN_MSEs
        # json.dump(results, open(num_path + f"/MSEs_seed{seed}_query{query_type}_data{args.data_type}_repeat{int(args.repeat*100)}percen.json", 'w'))
        json.dump(results, open(num_path + f"/MSEs_seed{seed}_query{query_type}_data{args.data_type}_noise{int(args.noise*100)}.json", 'w'))

        # save ssim
        # ssims = {}
        # ssims["PC"] = PC_SSIMs
        # ssims["HN"] = HN_SSIMs
        # print(ssims)
        # json.dump(ssims, open(num_path + f"/SSIMs_seed{seed}_query{query_type}_data{args.data_type}.json", 'w'))

if __name__ == "__main__":
    for s in args.seed:
        start_time = time.time()
        args.seed = s
        main(args)
        print(f'Seed complete, total time: {time.time() - start_time}')
