import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
from src.models import TemporalPC
from src.utils import *
from src.get_data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

result_path = os.path.join('./results/', 'moving_bar')
if not os.path.exists(result_path):
    os.makedirs(result_path)


seq_len = 5
h, w = 5, 5
inf_iters = 5 # for this experiment inf iters too large is harmful
inf_lr = 5e-3
learn_iters = 1500
learn_lr = 1e-2
latent_size = 5
control_size = 10
flattened_size = h*w
sparse_penal = 0
n_cued = 1 # number of cued images
assert(n_cued < seq_len)

X = torch.zeros((seq_len, h, w))

X[0, 0, :] = torch.ones((w))
X[1, 2, :] = torch.ones((w))
X[2, 4, :] = torch.ones((w))
X[3, 2, :] = torch.ones((w))
X[4, 0, :] = torch.ones((w))
xs = X.reshape((seq_len, h*w)).to(device)
us = torch.zeros((seq_len, control_size)).to(device)

torch.manual_seed(1)
model = TemporalPC(control_size, latent_size, flattened_size, nonlin='linear').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_lr)

train_losses = []
for learn_iter in range(learn_iters):
    loss = 0
    # prev_z = model.init_hidden(1).to(device)
    prev_z = torch.zeros((1, latent_size)).to(device)
    for k in range(seq_len):
        x, u = xs[k, :], us[k, :]
        optimizer.zero_grad()
        model.inference(inf_iters, inf_lr, x, u, prev_z)
        energy = model.update_grads(x, u, prev_z)
        energy.backward()
        optimizer.step()
        prev_z = model.z.clone().detach()
        loss += energy.item() / seq_len
    train_losses.append(loss)
    print(f'Iteration {learn_iter+1}, loss {loss}')


# retrieval
# prev_z = model.init_hidden(1).to(device)
prev_z = torch.zeros((1, latent_size)).to(device)

# collect the cues
cue = torch.zeros((seq_len, flattened_size))

# collect the retrievals
recall = torch.zeros((seq_len, flattened_size))

inf_iters = 2000
hiddens = []
for k in range(seq_len):
    x, u = xs[k, :], us[k, :] # [batch_size, 784]
    if k + 1 <= n_cued:
        model.inference(inf_iters, inf_lr, x, u, prev_z)
        prev_z = model.z.clone().detach()
        recall[k, :] = x.clone().detach()
        cue[k, :] = x.clone().detach()
    else:
        prev_z, pred_x = model(u, prev_z)
        recall[k, :] = pred_x
        cue[k, :] = torch.zeros_like(pred_x)
    hiddens.append(prev_z.clone().detach())

plt.figure()
plt.plot(train_losses)
plt.savefig(result_path + f'/losses_moving_bar', dpi=150)

fig, ax = plt.subplots(2, 5)
for i in range(seq_len):
    ax[0, i].imshow(to_np(X[i]), cmap='gray')
    # ax[0, i].axis('off')  
    ax[0, i].set_xticks([])
    ax[0, i].set_yticks([])
    ax[1, i].imshow(to_np(recall[i].reshape((h, w))), cmap='gray')
    # ax[1, i].axis('off')
    ax[1, i].set_xticks([])
    ax[1, i].set_yticks([])
plt.savefig(result_path + '/moving_bar.pdf')

fig, ax = plt.subplots(1, 5, figsize=(6, 1.5), sharey=True)
for i in range(seq_len):
    ax[i].stem(to_np(hiddens[i]).reshape((latent_size)), linefmt='k', markerfmt='ok', basefmt = 'r')
    ax[i].set_xticks([])
    ax[i].set_frame_on(False)
    if i != 0:
        ax[i].tick_params(left=False)
plt.tight_layout()
plt.savefig(result_path + '/moving_bar_hidden.pdf')

img3 = to_np(model(us[1, :], hiddens[1])[1]).reshape((h, w))
img5 = to_np(model(us[3, :], hiddens[3])[1]).reshape((h, w))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img3)
ax[0].axis('off')
ax[1].imshow(img5)
ax[1].axis('off')
plt.savefig(result_path + '/forwarded.pdf')
