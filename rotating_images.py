import os
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'image_rotation')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# hyper parameters
sample_size = 1000
test_size = 1
batch_size = 500
seq_len = 10
inf_iters = 100
inf_lr = 1e-2
learn_iters = 100
learn_lr = 1e-4
latent_size = 256
control_size = 10
sparse_penal = 0
flattened_size = 784
n_cued = 1 # number of cued images
assert(n_cued < seq_len)
seed = 4
angle = 20

# load data
loader, test_data = get_rotating_mnist('./data', 
                                        seq_len, 
                                        sample_size,
                                        test_size,
                                        batch_size,
                                        seed, 
                                        device, 
                                        angle=angle, 
                                        test_digit=5)

model = TemporalPC(control_size, latent_size, flattened_size, nonlin='linear').to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learn_lr)

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

    print(f'Iteration {learn_iter+1}, loss {epoch_loss}, time {time.time()-epoch_start_time} seconds')
    losses.append(epoch_loss)

# cued prediction/inference
"""Should try to adapt this to batched test data!!!"""
print('Cued inference begins')
inf_iters = 5000

us = torch.zeros((test_size, seq_len, control_size)).to(device)
test_data = test_data.reshape((test_size, n_cued, flattened_size)).to(device)
prev_z = model.init_hidden(test_size).to(device)

# collect the cues
cue = torch.zeros((test_size, seq_len, flattened_size))

# collect the retrievals
recall = torch.zeros((test_size, seq_len, flattened_size))

for k in range(seq_len):
    u = us[:, k, :].squeeze() # [test_size, 784]
    if k + 1 <= n_cued:
        x = test_data[:, k, :]
        model.inference(inf_iters, inf_lr, x, u, prev_z)
        prev_z = model.z.clone().detach()
        recall[:, k, :] = x.clone().detach()
        cue[:, k, :] = x.clone().detach()
    else:
        prev_z, pred_x = model(u, prev_z)
        recall[:, k, :] = pred_x
        cue[:, k, :] = torch.zeros_like(pred_x)


# visualizations
n_mem = 5
d, _ = next(iter(loader))
print(d.shape)
fig, ax = plt.subplots(n_mem, seq_len, figsize=(seq_len, n_mem))
for i in range(n_mem):
    for j in range(seq_len):
        ax[i, j].imshow(to_np(d[i, j]).reshape(28, 28))
        ax[i, j].axis('off')
plt.savefig(result_path + f'/memory_size{sample_size}_len{seq_len}_learn{learn_iters}_auto', dpi=150)

plt.figure()
plt.plot(losses, label='squared error sum')
plt.legend()
plt.savefig(result_path + f'/rotation_losses_size{sample_size}_len{seq_len}_learn{learn_iters}_auto', dpi=150)

def plot_images(x, mode='recall'):
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, test_size))
    for j in range(seq_len):
        ax[j].imshow(to_np(x[j].squeeze().reshape(28, 28)))
        ax[j].axis('off')
    plt.savefig(result_path + f'/{mode}_size{sample_size}_len{seq_len}_learn{learn_iters}_auto', dpi=150)

plot_images(cue.squeeze(), mode='cue')
plot_images(recall.squeeze(), mode='recall')

Wr = to_np(model.Wr.weight)
plt.figure()
plt.imshow(Wr)
plt.title('Recurrent weight')
plt.grid(visible=None)
plt.colorbar()
plt.savefig(result_path + f'/Wr_size{sample_size}_len{seq_len}_learn{learn_iters}_auto', dpi=150)



