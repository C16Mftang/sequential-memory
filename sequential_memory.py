import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from src.models import TemporalPC
from src.utils import *
from src.get_data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'image_sequence')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# hyper parameters
seq_len = 10
sample_size = 2
batch_size = 2
inf_iters = 100
inf_lr = 1e-2
learn_iters = 1
learn_lr = 1e-4
latent_size = 256
control_size = 10
flattened_size = 784
sparse_penal = 0
n_cued = 1 # number of cued images
assert(n_cued < seq_len)

test_size = 1
angle = 20
seed = 1

# load data
loader = get_seq_mnist('./data', 
                       seq_len=seq_len,
                       sample_size=sample_size, 
                       batch_size=batch_size, 
                       seed=seed,
                       device=device)

# loader, test_data = get_rotating_mnist('./data', 
#                                         seq_len, 
#                                         sample_size,
#                                         test_size,
#                                         batch_size,
#                                         seed, 
#                                         device, 
#                                         angle=angle, 
#                                         test_digit=5)

model = TemporalPC(control_size, latent_size, flattened_size, nonlin='tanh').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

# training
losses = []
for learn_iter in range(learn_iters):
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

    print(f'Iteration {learn_iter+1}, loss {epoch_loss}')
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

def plot_images(x, mode='memory', show_size=sample_size):
    fig, ax = plt.subplots(show_size, seq_len, figsize=(seq_len, show_size))
    for i in range(show_size):
        for j in range(seq_len):
            ax[i, j].imshow(to_np(x[i, j].reshape(28, 28)), cmap='gray_r')
            ax[i, j].axis('off')
    plt.savefig(result_path + f'/{mode}_size{sample_size}_len{seq_len}_learn{learn_iters}', dpi=150)

plot_images(memory, mode='memory', show_size=2)
plot_images(cue, mode='cue', show_size=2)
plot_images(recall, mode='recall', show_size=2)

# fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
# for i in range(seq_len):
#     ax[i].imshow(hidden_states[i].reshape(int(np.sqrt(latent_size)), int(np.sqrt(latent_size))))
#     ax[i].axis('off')
# plt.savefig(result_path + f'/latent_activity_len{seq_len}_inf{inf_iters}_inf_pred', dpi=150)
