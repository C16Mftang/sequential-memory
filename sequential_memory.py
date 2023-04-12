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
from src.models import TemporalPC
from src.utils import *
from src.get_data import *

result_path = os.path.join('./results/', 'image_sequence')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# add parser as varaible of the main class
parser = argparse.ArgumentParser(description='Sequential memories')
parser.add_argument('--seq-len', type=int, default=[10], nargs='+', 
                    help='length of input for training (default: 10),\
                          can specify multiple values separated by whitespace')
parser.add_argument('--latent-size', type=int, default=256,
                    help='hidden size for training (default: 256)')
parser.add_argument('--input-size', type=int, default=10,
                    help='input size for training (default: 10)')
parser.add_argument('--output-size', type=int, default=784,
                    help='output size for training (default: 784)')
parser.add_argument('--sample-size', type=int, default=5,
                    help='input sample size for training (default: 5)')
parser.add_argument('--batch-size', type=int, default=5,
                    help='input batch size for training (default: 5)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for model init (default: 1)')
parser.add_argument('--data-seed', type=int, default=10,
                    help='seed for data sampling (default: 10)')
parser.add_argument('--nonlinearity', type=str, default='tanh',
                    help='nonlinear function used in the model')
parser.add_argument('--n-cued', type=int, default=1,
                    help='number of cued patterns when recall begins')
args = parser.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data and model
    seq_len = args.seq_len
    sample_size = args.sample_size
    batch_size = args.batch_size
    learn_iters = args.epochs
    latent_size = args.latent_size
    control_size = args.input_size
    flattened_size = args.output_size
    n_cued = args.n_cued # number of cued images
    assert(n_cued < seq_len)
    data_seed = args.data_seed
    seed = args.seed
    sparse_penal = 0

    #hyper parameters for tunning
    inf_iters = 100
    inf_lr = 1e-2
    learn_lr = 1e-4

    print(f'Training variables: seq_len:{seq_len}')

    # load data
    loader = get_seq_mnist('./data', 
                        seq_len=seq_len,
                        sample_size=sample_size, 
                        batch_size=batch_size, 
                        seed=data_seed,
                        device=device)
    
    torch.manual_seed(seed)
    model = TemporalPC(control_size, latent_size, flattened_size, nonlin='tanh').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

    # training
    losses = []
    for learn_iter in range(learn_iters):
        epoch_start_time = time.time()
        epoch_loss = 0
        for xs, _ in loader:
            us = torch.zeros((batch_size, seq_len, control_size)).to(device)
            xs = xs.reshape((batch_size, seq_len, -1)).to(device)
            prev_z = model.init_hidden(batch_size).to(device)
            batch_loss = 0
            for k in range(seq_len):
                x, u = xs[:, k:k+1, :].squeeze(), us[:, k:k+1, :].squeeze()
                optimizer.zero_grad()
                model.inference(inf_iters, inf_lr, x, u, prev_z)
                energy = model.update_grads(x, u, prev_z)
                energy.backward()
                optimizer.step()
                prev_z = model.z.clone().detach()

                # add up the loss value at each time step
                batch_loss += energy.item() / seq_len

            # add the loss in this batch
            epoch_loss += batch_loss / batch_size
        if (learn_iter + 1) % 10 == 0:
            print(f'Iteration {learn_iter+1}, loss {epoch_loss}, time {time.time()-epoch_start_time} seconds')
        losses.append(epoch_loss)

    # cued prediction/inference
    """can just extract the first image in each batch and initialize retrieval no need for a separate datset"""
    print('Cued inference begins')
    inf_iters = 100 # increase inf_iters

    memory, cue, recall = [], [], []
    for xs, _ in loader:
        us = torch.zeros((batch_size, seq_len, control_size)).to(device)
        xs = xs.reshape((batch_size, seq_len, -1)).to(device)
        prev_z = model.init_hidden(batch_size).to(device)

        # collect the original memory
        batch_memory = torch.zeros((batch_size, seq_len, flattened_size))

        # collect the cues
        batch_cue = torch.zeros((batch_size, seq_len, flattened_size))

        # collect the retrievals
        batch_recall = torch.zeros((batch_size, seq_len, flattened_size))

        for k in range(seq_len):
            x, u = xs[:, k, :], us[:, k, :] # [batch_size, 784]
            batch_memory[:, k, :] = x.clone().detach()
            if k + 1 <= n_cued:
                model.inference(inf_iters, inf_lr, x, u, prev_z)
                prev_z = model.z.clone().detach()
                batch_recall[:, k, :] = x.clone().detach()
                batch_cue[:, k, :] = x.clone().detach()
            else:
                prev_z, pred_x = model(u, prev_z)
                batch_recall[:, k, :] = pred_x
                batch_cue[:, k, :] = torch.zeros_like(pred_x)
        
        memory.append(batch_memory)
        cue.append(batch_cue)
        recall.append(batch_recall)

    memory = torch.cat(memory, dim=0)
    cue = torch.cat(cue, dim=0)
    recall = torch.cat(recall, dim=0)

    plt.figure()
    plt.plot(losses, label='squared error sum')
    plt.legend()
    plt.savefig(result_path + f'/losses_len{seq_len}_inf{inf_iters}', dpi=150)

    def _plot_images(x, mode='memory', show_size=sample_size):
        fig, ax = plt.subplots(show_size, seq_len, figsize=(seq_len, show_size))
        for i in range(show_size):
            for j in range(seq_len):
                ax[i, j].imshow(to_np(x[i, j].reshape(28, 28)), cmap='gray_r')
                ax[i, j].axis('off')
        plt.savefig(result_path + f'/{mode}_size{sample_size}_len{seq_len}_learn{learn_iters}', dpi=150)

    _plot_images(memory, mode='memory', show_size=2)
    _plot_images(cue, mode='cue', show_size=2)
    _plot_images(recall, mode='recall', show_size=2)

if __name__ == "__main__":
    for seq_len in args.seq_len:
        args.seq_len = seq_len
        main(args)
