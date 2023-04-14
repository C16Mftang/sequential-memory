import torch
import torch.nn as nn
import numpy as np
import time

def to_np(x):
    return x.cpu().detach().numpy()

def to_torch(x, device):
    return torch.from_numpy(x).to(device)

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class Linear(nn.Module):
    def forward(self, inp):
        return inp

    def deriv(self, inp):
        return torch.ones((1,)).to(inp.device)
    
def train_singlelayer_tPC(pc, optimizer, seq, learn_iters, device):
    seq_len = seq.shape[0]
    losses = []
    start_time = time.time()
    for learn_iter in range(learn_iters):
        epoch_loss = 0
        prev = pc.init_hidden(1).to(device)
        batch_loss = 0
        for k in range(seq_len):
            x = seq[k]
            optimizer.zero_grad()
            energy = pc.get_energy(x, prev)
            energy.backward()
            optimizer.step()
            prev = x.clone().detach()

            # add up the loss value at each time step
            epoch_loss += energy.item() / seq_len
        losses.append(epoch_loss)
        if (learn_iter + 1) % 10 == 0:
            print(f'Epoch {learn_iter+1}, loss {epoch_loss}')

    print(f'training PC complete, time: {time.time() - start_time}')
    return losses

def train_multilayer_tPC(model, optimizer, seq, learn_iters, inf_iters, inf_lr, device):
    """
    Function to train multi layer tPC
    """
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

def singlelayer_recall(model, seq, device, args):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    if args.query == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq[:-1])) if args.data_type == 'binary' else model(seq[:-1])
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            recall[k] = torch.sign(model(recall[k-1:k])) if args.data_type == 'binary' else model(recall[k-1:k]) # 1xN

    return recall

def multilayer_recall(model, seq, inf_iters, inf_lr, args, device):
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

def hn_recall(model, seq, device, args):
    """recall function for pc
    
    seq: PxN sequence
    mode: online or offline
    binary: true or false

    output: (P-1)xN recall of sequence (starting from the second step)
    """
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    if args.query == 'online':
        # recall using true image at each step
        recall[1:] = torch.sign(model(seq, seq[:-1])) if args.data_type == 'binary' else model(seq, seq[:-1]) # (P-1)xN
    else:
        # recall using predictions from previous step
        prev = seq[0].clone().detach() # 1xN
        for k in range(1, seq_len):
            # prev = torch.sign(model(seq, prev)) if binary else model(seq, prev) # 1xN
            recall[k] = torch.sign(model(seq, recall[k-1:k])) if args.data_type == 'binary' else model(seq, recall[k-1:k]) # 1xN

    return recall
