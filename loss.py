import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def loss_backward(y, t, T):
  if T.shape[0]==2 and T.shape[1]==2:
      T = T + 1e-4 * np.random.rand(T.shape[0], T.shape[1])

  num_classes = T.shape[0]
  T_inv = Variable(torch.from_numpy(np.linalg.inv(T))).float().cuda()
 
  # loss correction
  indices = torch.unsqueeze(t.data.cpu(),1)
  labels = torch.zeros(t.size()[0], num_classes).scatter_(1, indices, 1.) #onehot
  labels_aug = torch.matmul(Variable(labels).cuda(), T_inv)
  loss = torch.mean(torch.sum(-labels_aug*F.log_softmax(y, dim=1), dim=1))

  return loss


def loss_sigua_bc(y, t, T, gamma=0.01):
  beta = 0.0

  if T.shape[0]==2 and T.shape[1]==2:
      T = T + 1e-4 * np.random.rand(T.shape[0], T.shape[1])

  num_classes = T.shape[0]
  T_inv = Variable(torch.from_numpy(np.linalg.inv(T))).float().cuda()
 
  indices = torch.unsqueeze(t.data,1)
  labels = torch.zeros(t.size()[0], num_classes).cuda().scatter_(1,indices,1.)
  labels_aug = torch.matmul(Variable(labels), T_inv)
  cross_entropy = torch.sum(-labels_aug * F.log_softmax(y, dim=1), dim=1)
  indicator = Variable(torch.ones(t.size()[0])).cuda()
  indicator[(cross_entropy < beta).detach()] = -gamma
  cross_entropy = cross_entropy*(indicator.detach())
  loss = torch.mean(cross_entropy)

  return loss

def loss_sigua_sl(y, t, forget_rate, ind, noise_or_not, sigua_rate=1.0):
    loss = F.cross_entropy(y, t, reduce = False)
    ind_sorted = np.argsort(loss.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]# from small to big

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    num_forget = len(loss_sorted) - num_remember
    num_sigua = int(num_forget * sigua_rate)

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember].cpu()]])/float(num_remember)

    loss_small = loss_sorted[:num_remember]
    loss_big = loss_sorted[num_remember:]
    
    if num_sigua > 0:
        loss_sigua = loss_big[:num_sigua]
    else:
        loss_sigua = loss_big[:1]
    return torch.sum(loss_small)/num_remember, pure_ratio, torch.sum(loss_sigua)/len(loss_sigua)


