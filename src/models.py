import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.linalg import inv
from src.utils import *

class KalmanFilter(nn.Module):
    """Kalman filter

    x: observation layer
    z: hidden layer
    """
    def __init__(self, A, B, C, Q, R, latent_size) -> None:
        super().__init__()
        self.A = A.clone()
        self.B = B.clone()
        self.C = C.clone()
        # control input, a list/1d array
        self.latent_size = latent_size
        # covariance matrix of noise
        self.Q = Q
        self.R = R
        
    def projection(self):
        z_proj = torch.matmul(self.A, self.z) + torch.matmul(self.B, self.u)
        P_proj = torch.matmul(self.A, torch.matmul(self.P, self.A.t())) + self.Q
        return z_proj, P_proj

    def correction(self, z_proj, P_proj):
        """Correction step in KF

        K: Kalman gain
        """
        K = torch.matmul(torch.matmul(P_proj, self.C.t()), inv(torch.matmul(torch.matmul(self.C, P_proj), self.C.t()) + self.R))
        self.z = z_proj + torch.matmul(K, self.x - torch.matmul(self.C, z_proj))
        self.P = P_proj - torch.matmul(K, torch.matmul(self.C, P_proj))

    def inference(self, inputs, controls):
        zs = []
        pred_xs = []
        exs = []
        seq_len = inputs.shape[1]
        # initialize mean and covariance estimates of the latent state
        self.z = torch.zeros((self.latent_size, 1)).to(inputs.device)
        self.P = torch.eye(self.latent_size).to(inputs.device)
        for l in range(seq_len):
            self.x = inputs[:, l:l+1]
            self.u = controls[:, l:l+1]
            z_proj, P_proj = self.projection()
            self.correction(z_proj, P_proj)
            zs.append(self.z.detach().clone())
            pred_x = torch.matmul(self.C, z_proj)
            pred_xs.append(pred_x)
            exs.append(self.x - pred_x)
        # collect predictions on the observaiton level
        pred_xs = torch.cat(pred_xs, dim=1)
        self.exs = torch.cat(exs, dim=1)
        zs = torch.cat(zs, dim=1)
        return zs, pred_xs
    
class MultilayertPC(nn.Module):
    """Multi-layer tPC class, using autograd"""
    def __init__(self, hidden_size, output_size, nonlin='tanh'):
        super(MultilayertPC, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
    
    def forward(self, prev_z):
        pred_z = self.Wr(self.nonlin(prev_z))
        pred_x = self.Wout(self.nonlin(pred_z))
        return pred_z, pred_x

    def init_hidden(self, bsz):
        """Initializing prev_z"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

    def update_errs(self, x, prev_z):
        pred_z, _ = self.forward(prev_z)
        pred_x = self.Wout(self.nonlin(self.z))
        err_z = self.z - pred_z
        err_x = x - pred_x
        return err_z, err_x
    
    def update_nodes(self, x, prev_z, inf_lr, update_x=False):
        err_z, err_x = self.update_errs(x, prev_z)
        delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
        self.z -= inf_lr * delta_z
        if update_x:
            delta_x = err_x
            x -= inf_lr * delta_x

    def inference(self, inf_iters, inf_lr, x, prev_z, update_x=False):
        """prev_z should be set up outside the inference, from the previous timestep

        Args:
            train: determines whether we are at the training or inference stage
        
        After every time step, we change prev_z to self.z
        """
        with torch.no_grad():
            # initialize the current hidden state with a forward pass
            self.z, _ = self.forward(prev_z)

            # update the values nodes
            for i in range(inf_iters):
                self.update_nodes(x, prev_z, inf_lr, update_x)
                
    def update_grads(self, x, prev_z):
        """x: input at a particular timestep in stimulus
        
        Could add some sparse penalty to weights
        """
        err_z, err_x = self.update_errs(x, prev_z)
        self.hidden_loss = torch.sum(err_z**2)
        self.obs_loss = torch.sum(err_x**2)
        energy = self.hidden_loss + self.obs_loss
        return energy
    

class SingleLayertPC(nn.Module):
    """Generic single layer tPC"""
    def __init__(self, input_size, nonlin='tanh'):
        super(SingleLayertPC, self).__init__()
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
        
        self.input_size = input_size
        
    def init_hidden(self, bsz):
        """Initializing sequence"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.input_size))
    
    def forward(self, prev):
        pred = self.Wr(self.nonlin(prev))
        return pred

    def update_errs(self, curr, prev):
        """
        curr: current observation
        prev: previous observation
        """
        pred = self.forward(prev)
        err = curr - pred
        return err
    
    def get_energy(self, curr, prev):
        err = self.update_errs(curr, prev)
        energy = torch.sum(err**2)
        return energy

class LinearSingleLayertPC(nn.Module):
    """
    Linear version of the single layer tPC;

    This is for the convenience of searching Pmax for binary patterns

    Training is performed across the whole sequence,
    rather than step-by-step i.e., the loss is the sum over
    all timesteps.
    """
    def __init__(self, input_size, learn_iters=100, lr=1e-2):
        super(LinearSingleLayertPC, self).__init__()
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        self.input_size = input_size
        self.learn_iters = learn_iters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(list(self.Wr.parameters()), lr=lr)
    
    def forward(self, s):
        pred = self.Wr(s)
        return pred
    
    def recall(self, s):
        pred = self.Wr(s)
        return pred
    
    def get_loss(self, X):
        """X: shape PxN"""
        pred = self.forward(X[:-1]) # (P-1)xN
        loss = self.criterion(pred, X[1:])
        return loss

    def train(self, X):
        losses = []
        for i in range(self.learn_iters):
            self.optimizer.zero_grad()
            loss = self.get_loss(X)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses


class AsymmetricHopfieldNetwork(nn.Module):
    
    def __init__(self, input_size):
        super(AsymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        
    def forward(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons
        output: PxN matrix
        """
        output = torch.sign(torch.matmul(X, self.W.t()))
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Asymmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[1:].T, X[:-1]) / N

class ModernAsymmetricHopfieldNetwork(nn.Module):
    """
    MAHN. train() function is simply a placeholder since we don't really train these models
    """
    
    def __init__(self, input_size, sep='linear', beta=1):
        super(ModernAsymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        self.sep = sep
        self.beta = beta
        
    def forward(self, X, s):
        """
        X: stored memories, shape PxN
        s: query, shape (P-1)xN
        output: (P-1)xN matrix
        """
        _, N = X.shape
        if self.sep == 'exp':
            score = torch.exp(torch.matmul(s, X[:-1].t()))
        elif self.sep == 'softmax':
            score = F.softmax(self.beta * torch.matmul(s, X[:-1].t()), dim=1)
        else:
            score = torch.matmul(s, X[:-1].t()) ** int(self.sep)
        output = torch.matmul(score, X[1:])
        
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Asymmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[1:].T, X[:-1]) / N

        return -1
                

# class TemporalPC(nn.Module):
#     def __init__(self, control_size, hidden_size, output_size, nonlin='tanh'):
#         """A more concise and pytorchy way of implementing tPC

#         Suitable for image sequences
#         """
#         super(TemporalPC, self).__init__()
#         self.hidden_size = hidden_size
#         self.Win = nn.Linear(control_size, hidden_size, bias=False)
#         self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.Wout = nn.Linear(hidden_size, output_size, bias=False)

#         if nonlin == 'linear':
#             self.nonlin = Linear()
#         elif nonlin == 'tanh':
#             self.nonlin = Tanh()
#         else:
#             raise ValueError("no such nonlinearity!")
    
#     def forward(self, u, prev_z):
#         pred_z = self.Win(self.nonlin(u)) + self.Wr(self.nonlin(prev_z))
#         pred_x = self.Wout(self.nonlin(pred_z))
#         return pred_z, pred_x

#     def init_hidden(self, bsz):
#         """Initializing prev_z"""
#         return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

#     def update_errs(self, x, u, prev_z):
#         pred_z, _ = self.forward(u, prev_z)
#         pred_x = self.Wout(self.nonlin(self.z))
#         err_z = self.z - pred_z
#         err_x = x - pred_x
#         return err_z, err_x
    
#     def update_nodes(self, x, u, prev_z, inf_lr, update_x=False):
#         err_z, err_x = self.update_errs(x, u, prev_z)
#         delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
#         self.z -= inf_lr * delta_z
#         if update_x:
#             delta_x = err_x
#             x -= inf_lr * delta_x

#     def inference(self, inf_iters, inf_lr, x, u, prev_z, update_x=False):
#         """prev_z should be set up outside the inference, from the previous timestep

#         Args:
#             train: determines whether we are at the training or inference stage
        
#         After every time step, we change prev_z to self.z
#         """
#         with torch.no_grad():
#             # initialize the current hidden state with a forward pass
#             self.z, _ = self.forward(u, prev_z)

#             # update the values nodes
#             for i in range(inf_iters):
#                 self.update_nodes(x, u, prev_z, inf_lr, update_x)
                
#     def update_grads(self, x, u, prev_z):
#         """x: input at a particular timestep in stimulus
        
#         Could add some sparse penalty to weights
#         """
#         err_z, err_x = self.update_errs(x, u, prev_z)
#         self.hidden_loss = torch.sum(err_z**2)
#         self.obs_loss = torch.sum(err_x**2)
#         energy = self.hidden_loss + self.obs_loss
#         return energy