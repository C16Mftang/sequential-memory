import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.linalg import inv
from src.utils import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.utils import *
from src.get_data import *
from src.models import SingleLayertPC

result_path = os.path.join('./results/', 'single_layer')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# add parser as varaible of the main class
parser = argparse.ArgumentParser(description='Sequential memories')
parser.add_argument('--seq-len', type=int, default=[10], nargs='+', 
                    help='length of input for training (default: 10),\
                          can specify multiple values separated by whitespace')
parser.add_argument('--output-size', type=int, default=784,
                    help='output size for training (default: 784)')
parser.add_argument('--sample-size', type=int, default=2,
                    help='input sample size for training (default: 5)')
parser.add_argument('--batch-size', type=int, default=2,
                    help='input batch size for training (default: 5)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
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
    flattened_size = args.output_size
    n_cued = args.n_cued # number of cued images
    assert(n_cued < seq_len)
    data_seed = args.data_seed
    seed = args.seed

    #hyper parameters for tunning later
    learn_lr = 2e-4

    print(f'Training variables: seq_len:{seq_len}')

    # load data
    loader = get_seq_mnist('./data', 
                        seq_len=seq_len,
                        sample_size=sample_size, 
                        batch_size=batch_size, 
                        seed=data_seed,
                        device=device)
    
    # initialize the model
    torch.manual_seed(seed)
    model = SingleLayertPC(flattened_size, nonlin='tanh').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

    # training
    losses = []
    for learn_iter in range(learn_iters):
        epoch_start_time = time.time()
        epoch_loss = 0
        for xs, _ in loader:
            xs = xs.reshape((batch_size, seq_len, -1)).to(device)
            prev = model.init_hidden(batch_size).to(device)
            batch_loss = 0
            for k in range(seq_len):
                x = xs[:, k, :].squeeze()

                # training
                optimizer.zero_grad()
                energy = model.get_energy(x, prev)
                energy.backward()
                optimizer.step()

                # reload the prev with current value
                prev = x.clone().detach()

                # add up the loss value at each time step
                batch_loss += energy.item() / seq_len

            # add the loss in this batch
            epoch_loss += batch_loss / batch_size

        print(f'Iteration {learn_iter+1}, loss {epoch_loss}, time {time.time()-epoch_start_time} seconds')
        losses.append(epoch_loss)

    # cued prediction/inference
    print('Cued inference begins')

    memory, cue, recall = [], [], []
    for xs, _ in loader:
        xs = xs.reshape((batch_size, seq_len, -1)).to(device)
        prev = model.init_hidden(batch_size).to(device)

        # collect the original memory
        batch_memory = torch.zeros((batch_size, seq_len, flattened_size))

        # collect the cues
        batch_cue = torch.zeros((batch_size, seq_len, flattened_size))

        # collect the retrievals
        batch_recall = torch.zeros((batch_size, seq_len, flattened_size))

        for k in range(seq_len):
            x = xs[:, k, :] # [batch_size, 784]
            batch_memory[:, k, :] = x.clone().detach()
            if k + 1 <= n_cued:
                prev = x.clone().detach()

                # dump the cue into recall and cue lists
                batch_recall[:, k, :] = x.clone().detach()
                batch_cue[:, k, :] = x.clone().detach()
            else:
                prev = model(prev)

                # dump the prediction into recall
                batch_recall[:, k, :] = prev

                # dump a zero vector to cue
                batch_cue[:, k, :] = torch.zeros_like(prev)
        
        memory.append(batch_memory)
        cue.append(batch_cue)
        recall.append(batch_recall)

    memory = torch.cat(memory, dim=0)
    cue = torch.cat(cue, dim=0)
    recall = torch.cat(recall, dim=0)

    plt.figure()
    plt.plot(losses, label='squared error sum')
    plt.legend()
    plt.savefig(result_path + f'/losses_len{seq_len}', dpi=100)

    def _plot_images(x, mode='memory', show_size=sample_size):
        fig, ax = plt.subplots(show_size, seq_len, figsize=(seq_len, show_size))
        for i in range(show_size):
            for j in range(seq_len):
                ax[i, j].imshow(to_np(x[i, j].reshape(28, 28)), cmap='gray_r')
                ax[i, j].axis('off')
        plt.savefig(result_path + f'/{mode}_size{sample_size}_len{seq_len}_learn{learn_iters}', dpi=100)

    _plot_images(memory, mode='memory', show_size=2)
    _plot_images(cue, mode='cue', show_size=2)
    _plot_images(recall, mode='recall', show_size=2)

if __name__ == "__main__":
    for seq_len in args.seq_len:
        args.seq_len = seq_len
        main(args)