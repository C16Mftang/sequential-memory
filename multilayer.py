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
from src.models import TemporalPC, MultilayertPC
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
                    help='hidden size for training (default: 256)')
parser.add_argument('--input-size', type=int, default=784,
                    help='input size for training (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--nonlinearity', type=str, default='tanh',
                    help='nonlinear function used in the model')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'recall'],
                    help='mode of the script: train or recall (just to save time)')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
args = parser.parse_args()


def train_PC(model, optimizer, seq, learn_iters, inf_iters, inf_lr, device):
    seq_len = seq.shape[0]
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        prev_z = model.init_hidden(1).to(device)
        for k in range(seq_len):
            x = seq[k].clone().detach()
            optimizer.zero_grad()
            model.inference(inf_iters, inf_lr, x, prev_z)
            energy = model.update_grads(x, prev_z)
            energy.backward()
            optimizer.step()
            prev_z = model.z.clone().detach()

            # add up the loss value at each time step
            epoch_loss += energy.item() / seq_len

        losses.append(epoch_loss)
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')
        
    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def _recall(model, seq, inf_iters, inf_lr, args, device):
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    prev_z = model.init_hidden(1).to(device)

    if args.query == 'online':
        # infer the latent state at each time step, given correct previous input
        for k in range(seq_len-1):
            x = seq[k].clone().detach()
            model.inference(inf_iters, inf_lr, x, prev_z)
            prev_z = model.z.clone().detach()
            _, pred_x = model(prev_z)
            recall[k+1] = pred_x

    elif args.query == 'offline':
        # only infer the latent of the cue, then forward pass
        x = seq[0].clone().detach()
        model.inference(inf_iters, inf_lr, x, prev_z)
        prev_z = model.z.clone().detach()

        # fast forward pass
        for k in range(1, seq_len):
            prev_z, pred_x = model(prev_z)
            recall[k] = pred_x

    return recall

def _plot_recalls(recall, args):
    seq_len = recall.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(recall[j].reshape(28, 28)), cmap='gray_r')
        ax[j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/mtPC_len{seq_len}_query{args.query}')

def _plot_memory(x):
    seq_len = x.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        ax[j].imshow(to_np(x[j].reshape(28, 28)), cmap='gray_r')
        ax[j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/memory_len{seq_len}')

def _plot_PC_loss(loss, seq_len, learn_iters):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/losses_len{seq_len}_iters{learn_iters}')
        

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data and model
    seq_len_max = args.seq_len_max
    learn_iters = args.epochs
    learn_lr = args.lr
    latent_size = args.latent_size
    input_size = args.input_size
    seed = args.seed
    mode = args.mode
    
    # inference variables: no need to tune too much
    inf_iters = 100
    inf_lr = 1e-2

    MSEs = []
    seq_lens = [2 ** pow for pow in range(1, seq_len_max)]
    for seq_len in seq_lens:
        print(f'Training variables: seq_len:{seq_len}; seed:{seed}')

        # load data
        seq = load_sequence_mnist(seed, seq_len, order=False).to(device)
        seq = seq.reshape((seq_len, input_size)) # seq_lenx784
        
        # multilayer PC
        model = MultilayertPC(latent_size, input_size, nonlin='tanh').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

        # path to save model to/load model from
        PATH = os.path.join(model_path, f'PC_len{seq_len}_seed{seed}.pt')

        if mode == 'train':
            # train PC
            PC_losses = train_PC(model, optimizer, seq, learn_iters, inf_iters, inf_lr, device)
            # save the current model and plot the loss for tunning
            torch.save(model.state_dict(), PATH)
            _plot_PC_loss(PC_losses, seq_len, learn_iters)
        
        elif mode == 'recall':
            # recall mode, no training need, fast
            model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
            model.eval()

            with torch.no_grad():
                recalls = _recall(model, seq, inf_iters, inf_lr, args, device)

            if seq_len <= 16:
                _plot_recalls(recalls, args)
                _plot_memory(seq)
            
            MSEs.append(float(to_np(torch.mean((seq - recalls) ** 2))))

    if mode == 'recall':
        results = {}
        results["PC"] = MSEs
        json.dump(results, open(num_path + f"/MSEs_seed{seed}_query{args.query}.json", 'w'))

if __name__ == "__main__":
    for s in args.seed:
        args.seed = s
        main(args)
