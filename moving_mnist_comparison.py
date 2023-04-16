import os
import json
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from src.models import TemporalPC, MultilayertPC, SingleLayertPC, ModernAsymmetricHopfieldNetwork
from src.utils import *
from src.get_data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

path = 'moving_mnist'
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

# training parameters as command line arguments
parser = argparse.ArgumentParser(description='Generalization capabilities')

parser.add_argument('--seed', type=int, default=[1], nargs='+',
                    help='seed for model init (default: 1); can be multiple, separated by space')
parser.add_argument('--sample-size-max', type=int, default=110, 
                    help='max number of sequences with motion')
parser.add_argument('--batch-size', type=int, default=100, 
                    help='training batch size')
parser.add_argument('--test-size', type=int, default=5, 
                    help='number of unseen sequences with motion for testing')
parser.add_argument('--input-size', type=int, default=1024,
                    help='input size for training (default: 1024)')
parser.add_argument('--latent-size', type=int, default=630,
                    help='hidden size for training (default: 630)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train (default: 600)')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--mode', type=str, default='train single', choices=['train single', 'recall', 'train multi'],
                    help='mode of the script: train or recall or generalization')
parser.add_argument('--nonlinearity', type=str, default='linear',
                    help='nonlinear function used in the model')
parser.add_argument('--data-type', type=str, default='continuous', choices=['binary', 'continuous'],
                    help='type of data; note that when HN type is exp or softmax, \
                        this should be always continuous')
parser.add_argument('--beta', type=int, default=1,
                    help='beta value for the MCHN')
args = parser.parse_args()


def train_batched_input(model, optimizer, loader, learn_iters, inf_iters, inf_lr, device, nlayer=2):
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        for xs in loader:
            xs = xs[0]
            batch_size, seq_len = xs.shape[:2]

            # reshape image to vector
            xs = xs.reshape((batch_size, seq_len, -1)).to(device)

            # initialize the hidden activities
            prev = model.init_hidden(batch_size).to(device)

            if nlayer > 1: # train multilayer tPC
                batch_loss = 0
                for k in range(seq_len):
                    x = xs[:, k, :].clone().detach()
                    optimizer.zero_grad()
                    model.inference(inf_iters, inf_lr, x, prev)
                    energy = model.update_grads(x, prev)
                    energy.backward()
                    optimizer.step()
                    prev = model.z.clone().detach()

                    # add up the loss value at each time step
                    batch_loss += energy.item() / seq_len

                # add the loss in this batch
                epoch_loss += batch_loss / batch_size

            else:
                batch_loss = 0
                for k in range(seq_len):
                    x = xs[:, k, :].clone().detach()
                    optimizer.zero_grad()
                    energy = model.get_energy(x, prev)
                    energy.backward()
                    optimizer.step()
                    prev = x.clone().detach()

                    # add up the loss value at each time step
                    batch_loss += energy.item() / seq_len

                # add the loss in this batch
                epoch_loss += batch_loss / batch_size

        losses.append(epoch_loss)
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')

    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def _pc_recall(model, seq, inf_iters, inf_lr, args, device, nlayer=2):
    seq_len, N = seq.shape

    # initialize the recalls tensor
    recall = torch.zeros_like(seq).to(device)
    recall[0] = seq[0].clone().detach()

    # initialize the hidden activities
    prev = model.init_hidden(1).to(device)

    if nlayer > 1: # multilayer models
        if args.query == 'online':
            # infer the latent state at each time step, given correct previous input
            for k in range(seq_len-1):
                x = seq[k].clone().detach()
                model.inference(inf_iters, inf_lr, x, prev)
                prev = model.z.clone().detach()
                _, pred_x = model(prev)
                recall[k+1] = pred_x

        elif args.query == 'offline':
            # only infer the latent of the cue, then forward pass
            x = seq[0].clone().detach()
            model.inference(inf_iters, inf_lr, x, prev)
            prev = model.z.clone().detach()

            # fast forward pass
            for k in range(1, seq_len):
                prev, pred_x = model(prev)
                recall[k] = pred_x
    
    else: # single layer models
        if args.query == 'online':
            # recall using true image at each step
            recall[1:] = model(seq[:-1])
        else:
            # recall using predictions from previous step
            prev = seq[0].clone().detach() # bszxN
            for k in range(1, seq_len):
                recall[k] = model(recall[k-1]) # bszxN
    
    # return mse and the final batch's recall for plotting
    return recall

def _hn_recall(model, seq, X, device, args):
    """
    Basically, we compare the frames in seq with all frames in all sequences

    X: sample_size x seq_len x N, the whole memory
    seq: seq_len x N
    """
    sample_size, seq_len, N = X.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()

    # select all frames from all squences, except the final frames
    # this is our "key"
    K = X[:, :-1].reshape((sample_size * (seq_len - 1), N))

    # select all frames from all sequences except the first frames
    # this is out "value"
    V = X[:, 1:].reshape((sample_size * (seq_len - 1), N))

    if args.query == 'online':
        # recall using true image at each step
        # recall[1:] = model(X, seq[:-1]) # (P-1)xN
        score = F.softmax(args.beta * torch.matmul(seq[:-1], K.t()), dim=1) # (seq_len-1) x (sample_size)(seq_len-1)
        recall[1:] = torch.matmul(score, V) # (P-1)xN
    else:
        # recall using predictions from previous step
        for k in range(1, seq_len):
            score = F.softmax(args.beta * torch.matmul(recall[k-1:k], K.t()), dim=1) # 1 x (sample_size)(seq_len-1)
            recall[k] = torch.matmul(score, V) # 1xN

    return recall

def _plot_PC_loss(loss, sample_size, learn_iters, name):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/{name}_losses_size{sample_size}_iters{learn_iters}')

def _plot_recalls(recall, test_size, args, name, sample_size):
    seq_len = recall.shape[1]
    fig, ax = plt.subplots(test_size, seq_len, figsize=(seq_len, test_size))
    for i in range(test_size):
        for j in range(seq_len):
            ax[i, j].imshow(to_np(recall[i, j].reshape(32, 32)), cmap='gray_r')
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/{name}_size{sample_size}_{args.query}', dpi=200)

def _plot_memory(x, test_size, args, sample_size):
    seq_len = x.shape[1]
    fig, ax = plt.subplots(test_size, seq_len, figsize=(seq_len, test_size))
    for i in range(test_size):
        for j in range(seq_len):
            ax[i, j].imshow(to_np(x[i, j].reshape(32, 32)), cmap='gray_r')
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(fig_path + f'/memory_size{sample_size}', dpi=200)

def main(args):
    # hyper parameters
    seq_len = 10
    sample_size_max = args.sample_size_max
    test_size = args.test_size
    learn_iters = args.epochs
    learn_lr = args.lr
    latent_size = args.latent_size
    input_size = args.input_size
    seed = args.seed
    nonlin = args.nonlinearity
    mode = args.mode

    # fix these
    inf_iters = 100
    inf_lr = 1e-2

    # recall MSEs
    sPC_MSEs, mPC_MSEs, HN_MSEs = [], [], []

    for sample_size in range(10, sample_size_max, 10):
        batch_size = sample_size

        print(f'Training variables: size:{sample_size}; seed:{seed}')

        # load data
        loader = get_moving_mnist('./data/mnist', sample_size, batch_size, seed)

        # singlelayer PC
        spc = SingleLayertPC(input_size, nonlin=nonlin).to(device)
        s_optimizer = torch.optim.Adam(spc.parameters(), lr=learn_lr)

        # multilayer PC
        mpc = MultilayertPC(latent_size, input_size, nonlin=nonlin).to(device)
        m_optimizer = torch.optim.Adam(mpc.parameters(), lr=learn_lr)

        # MCHN 
        hn = ModernAsymmetricHopfieldNetwork(input_size, sep='softmax', beta=5).to(device)

        PATH = os.path.join(model_path, f'PC_rotation_size{sample_size}_nonlin{nonlin}.pt')
        if mode == 'train single':
            # training single layer tPC
            sPC_losses = train_batched_input(spc, s_optimizer, loader, learn_iters, inf_iters, inf_lr, device, nlayer=1)
            torch.save(spc.state_dict(), os.path.join(model_path, f'sPC_size{sample_size}_seed{seed}.pt'))
            _plot_PC_loss(sPC_losses, sample_size, learn_iters, "sPC")

        elif mode == 'train multi':
            # training 2-layer tPC
            mPC_losses = train_batched_input(mpc, m_optimizer, loader, learn_iters, inf_iters, inf_lr, device, nlayer=2)
            torch.save(mpc.state_dict(), os.path.join(model_path, f'mPC_size{sample_size}_seed{seed}.pt'))
            _plot_PC_loss(mPC_losses, sample_size, learn_iters, "mPC")

        elif mode == 'recall':
            # spc
            spc.load_state_dict(torch.load(os.path.join(model_path, f'sPC_size{sample_size}_seed{seed}.pt'), 
                                        map_location=torch.device(device)))
            spc.eval()

            # mpc
            mpc.load_state_dict(torch.load(os.path.join(model_path, f'mPC_size{sample_size}_seed{seed}.pt'), 
                                           map_location=torch.device(device)))
            mpc.eval()


            # load the whole dataset 
            memories = []
            for xs, _ in loader:
                # xs: bszxseq_lenxN
                for i in range(batch_size):
                    x = xs[i].reshape((seq_len, input_size))
                    memories.append(x)
            memories = torch.stack(memories, dim=0).to(device) # sample_size, seq_len, 1024

            # initialize recalls
            s_recalls = torch.zeros_like(memories)
            m_recalls = torch.zeros_like(memories)
            hn_recalls = torch.zeros_like(memories)

            with torch.no_grad():
                for i in range(sample_size):
                    memory = memories[i]
                    s_recalls[i] = _pc_recall(spc, memory, inf_iters, inf_lr, args, device, nlayer=1)
                    m_recalls[i] = _pc_recall(mpc, memory, inf_iters, inf_lr, args, device, nlayer=2)
                    hn_recalls[i] = _hn_recall(hn, memory, memories, device, args)

            if sample_size == 20:
                _plot_recalls(s_recalls, test_size, args, 'spc', sample_size)
                _plot_recalls(m_recalls, test_size, args, 'mpc', sample_size)
                _plot_recalls(hn_recalls, test_size, args, 'hn', sample_size)
                _plot_memory(memories, test_size, args, sample_size)

            sPC_MSEs.append(float(to_np(torch.mean((memories - s_recalls) ** 2))))
            HN_MSEs.append(float(to_np(torch.mean((memories - hn_recalls) ** 2))))
        
    # save everything at this particular seed
    if mode == 'recall':
        results = {}
        results["sPC"] = sPC_MSEs
        results["mPC"] = mPC_MSEs
        results["HN"] = HN_MSEs
        json.dump(results, open(num_path + f"/MSEs_seed{seed}_query{args.query}.json", 'w'))
            


if __name__ == "__main__":
    for s in args.seed:
        start_time = time.time()
        args.seed = s
        main(args)
        print(f'{args.mode} finishes, total time elapsed:{time.time() - start_time}')



