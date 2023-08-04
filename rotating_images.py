import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from src.models import MultilayertPC
from src.utils import *
from src.get_data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

path = 'generalization'
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
parser.add_argument('--sample-size', type=int, default=1000, 
                    help='number of sequences with motion')
parser.add_argument('--test-size', type=int, default=50, 
                    help='number of unseen sequences with motion for testing')
parser.add_argument('--batch-size', type=int, default=500, 
                    help='training batch size')
parser.add_argument('--input-size', type=int, default=784,
                    help='input size for training (default: 784)')
parser.add_argument('--latent-size', type=int, default=480,
                    help='hidden size for training (default: 480)')
parser.add_argument('--seed', type=int, default=[1], nargs='+',
                    help='seed for model init and data sampling')
parser.add_argument('--angle', type=int, default=20,
                    help='rotating angles for the rotational experiments')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--query', type=str, default='offline', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'recall', 'generalize', 'PCA'],
                    help='mode of the script: train or recall or generalization')
parser.add_argument('--nonlinearity', type=str, default='tanh',
                    help='nonlinear function used in the model')
parser.add_argument('--dynamic', type=str, default='rotation', choices=['rotation', 'bouncing'],
                    help='type of dynamics')
args = parser.parse_args()


def train_batched_input(model, optimizer, loader, learn_iters, inf_iters, inf_lr, device):
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
            prev_z = model.init_hidden(batch_size).to(device)

            batch_loss = 0
            for k in range(seq_len):
                x = xs[:, k, :].clone().detach()
                optimizer.zero_grad()
                model.inference(inf_iters, inf_lr, x, prev_z)
                energy = model.update_grads(x, prev_z)
                energy.backward()
                optimizer.step()
                prev_z = model.z.clone().detach()

                # add up the loss value at each time step
                batch_loss += energy.item() / seq_len

            # add the loss in this batch
            epoch_loss += batch_loss / batch_size

        losses.append(epoch_loss)
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')

    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def _generalize(model, cue, seq_len, inf_iters, inf_lr, device):
    N = cue.shape[-1]
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = cue.clone().detach()
    prev_z = model.init_hidden(1).to(device)

    # only infer the latent of the cue, then forward pass
    x = cue.clone().detach()
    model.inference(inf_iters, inf_lr, x, prev_z)
    prev_z = model.z.clone().detach()

    # fast forward pass
    for k in range(1, seq_len):
        prev_z, pred_x = model(prev_z)
        recall[k] = pred_x

    return recall

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

def _plot_PC_loss(loss, sample_size, learn_iters):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/losses_size{sample_size}_iters{learn_iters}')

def _plot_recalls(recall, args):
    n_seq = 1
    seq_len = recall.shape[1]
    recall = recall.reshape((-1, 784))
    fig, ax = plt.subplots(n_seq, seq_len, figsize=(seq_len, n_seq))
    for i, a in enumerate(ax.flatten()):
        a.imshow(to_np(recall[i].reshape(28, 28)), cmap='gray_r')
        a.axis('off')
        a.set_xticklabels([])
        a.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fig_path + f'/{args.mode}_size{args.sample_size}_{args.query}_{args.dynamic}', dpi=150)

def _plot_gt(memory, args):
    n_seq = 1
    seq_len = memory.shape[1]
    memory = memory.reshape((-1, 784))
    fig, ax = plt.subplots(n_seq, seq_len, figsize=(seq_len, n_seq))
    for i, a in enumerate(ax.flatten()):
        a.imshow(to_np(memory[i].reshape(28, 28)), cmap='gray_r')
        a.axis('off')
        a.set_xticklabels([])
        a.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fig_path + f'/gt_{args.mode}_size{args.sample_size}_{args.query}_{args.dynamic}', dpi=150)

def main(args):
    # hyper parameters
    seq_len = 10
    sample_size = args.sample_size
    test_size = args.test_size
    batch_size = args.batch_size
    learn_iters = args.epochs
    learn_lr = args.lr
    latent_size = args.latent_size
    input_size = args.input_size
    seed = args.seed
    angle = args.angle
    nonlin = args.nonlinearity
    mode = args.mode

    # fix these
    inf_iters = 20
    inf_lr = 1e-2

    # load data
    loader = get_rotating_mnist('./data', seq_len, sample_size, batch_size, seed, angle)

    # prepare model
    model = MultilayertPC(latent_size, input_size, nonlin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

    PATH = os.path.join(model_path, f'PC_rotation_size{sample_size}_nonlin{nonlin}_seed{seed}.pt')
    if mode == 'train':
        # training
        PC_losses = train_batched_input(model, optimizer, loader, learn_iters, inf_iters, inf_lr, device)
        torch.save(model.state_dict(), PATH)

        _plot_PC_loss(PC_losses, sample_size, learn_iters)

    elif mode == 'generalize':
        model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
        model.eval()

        # load EMNIST
        test_data = load_sequence_emnist(48, test_size).to(device) # test_sizex28x28

        # rotate ground truth
        true_sequence = torch.zeros((test_size, seq_len, input_size))
        for l in range(seq_len):
            true_sequence[:, l] = TF.rotate(test_data, angle * l).reshape((-1, input_size))
        
        test_data = test_data.reshape((-1, input_size)) # test_sizex784

        # obtain generalization
        generalization = torch.zeros((test_size, seq_len, input_size))
        with torch.no_grad():
            for i in range(test_size):
                cue = test_data[i] # 1x784
                generalization[i] = _generalize(model, cue, seq_len, inf_iters, inf_lr, device)

        # visualize the generalization
        _plot_recalls(generalization, args)
        _plot_gt(true_sequence, args)

        # save MSE
        mse = to_np(torch.mean((true_sequence - generalization) ** 2))
        return mse

    elif mode == 'recall':
        # select a few examples from the training set to recall
        model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
        model.eval()

        test_data = next(iter(loader))[0][:test_size].to(device)
        test_data = test_data.reshape((test_size, seq_len, -1)) # test_size, seq_len, 784

        recalls = torch.zeros((test_size, seq_len, input_size)).to(device)
        with torch.no_grad():
            for i in range(test_size):
                seq = test_data[i]
                recalls[i] = _recall(model, seq, inf_iters, inf_lr, args, device)

        # visualize recall
        _plot_recalls(recalls, args)
        _plot_gt(test_data, args)

        # save MSE
        mse = to_np(torch.mean((test_data - recalls) ** 2))
        return mse

    elif mode == 'PCA':
        n_pcs = 3
        model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
        model.eval()

        test_data = next(iter(loader))[0].to(device) # bsz, seq, 28, 28
        test_data = test_data.reshape((batch_size, seq_len, -1)) # bsz, seq_len, 784

        pcs = np.zeros((batch_size, seq_len, n_pcs))

        prev_z = model.init_hidden(batch_size).to(device) # bsz, 480

        x = test_data[:, 0, :].clone().detach()
        model.inference(inf_iters, inf_lr, x, prev_z)
        prev_z = model.z.clone().detach().squeeze() # bsz, 480

        # PCA on the current step's hidden activity
        pca = PCA(n_components=n_pcs)
        pcs[:, 0, :] = pca.fit_transform(to_np(prev_z)) # bsz, 3

        for k in range(1, seq_len):
            prev_z, _ = model(prev_z) # bsz, 480
            # independent PCA on the current step's hidden activity
            pca = PCA(n_components=n_pcs)
            pcs[:, k, :] = pca.fit_transform(to_np(prev_z)) # bsz, 3

        example = pcs[1:4] # 5xseq_lenx3

        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(example.shape[0]):
            ax.plot(example[i, :, 0], example[i, :, 1], example[i, :, 2])

        plt.savefig(fig_path + '/PCA', dpi=150)
            
if __name__ == "__main__":
    mses = np.zeros((len(args.seed), 1))
    for ind, s in enumerate(args.seed):
        start_time = time.time()
        args.seed = s
        if args.mode != 'train':
            mses[ind] = main(args)
        else:
            main(args)
        print(f'Seed {s} complete, total time: {time.time() - start_time}')

    if args.mode != 'train':
        print(mses)
        # np.save(num_path + f'/{args.mode}_mses_{args.sample_size}', mses)


