# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys
import shutil
import datetime
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR 

from data.mnist import MNIST
from data.cifar import CIFAR10
from arguments import get_args
from loss import loss_sigua_sl, loss_sigua_bc, loss_backward

args=get_args()

def save_checkpoint(state, is_best, model_dir='', model_str=''):
    model_name = os.path.join(model_dir, model_str + '_chckpoint.pth.tar')
    best_model_name = os.path.join(model_dir, model_str + '_model_best.pth.tar')
    torch.save(state, model_name)
    if is_best:
        shutil.copyfile(model_name, best_model_name)

def resume_checkpoint(optimizer, model, model_dir='', model_str='', model_only_flag=False):
    model_name = os.path.join(model_dir, model_str + '_chckpoint.pth.tar')
    if os.path.isfile(model_name):
        print("==>loading checkpoint '{}'".format(model_name))
        checkpoint = torch.load(model_name)
        if not model_only_flag:
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==>load checkpoint '{}' (epoch {})".format(model_name, checkpoint['epoch']))
            return start_epoch, best_prec1, optimizer, model
        else:
          model.load_state_dict(checkpoint['state_dict'])
          return model 
    else:
        print("no checkpoint found at '{}'".format(model_name))

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                                )
    
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
 
noise_or_not = train_dataset.noise_or_not

# Train the Model
def train(train_loader, epoch, model, optimizer, args):
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    
    train_total=0
    train_correct=0 

    for i, (data, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        data = Variable(data).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits=model(data)
        prec1, = accuracy(logits, labels, topk=(1,))
        train_total+=1
        train_correct+=prec1
        # For SIGUA_SL
        if args.model_type=='sigua_sl':
            loss_l_small, pure_ratio, loss_l_big = \
              loss_sigua_sl(logits, labels, drop_rate_schedule[epoch], ind, noise_or_not, args.sigua_rate)
            loss= loss_l_small - args.sigua_scale*loss_l_big
        # For SIGUA_BC
        elif args.model_type=='sigua_bc':
            pure_ratio = 1 - args.noise_rate
            if epoch > args.warm_up:
                loss= loss_sigua_bc(logits, labels, train_dataset.P, args.sigua_scale)
            else:
                loss= loss_backward(logits, labels, train_dataset.P)
        else:
            loss= 0.0
            pure_ratio = 1 - args.noise_rate
              
        pure_ratio_list.append(100*pure_ratio)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Acc: %.4F, Loss: %.4f, Pure Ratio %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//args.batch_size, \
                    prec1, loss.item(), np.sum(pure_ratio_list)/len(pure_ratio_list)))
    train_acc=float(train_correct)/float(train_total)
    return train_acc, pure_ratio_list


# Evaluate the Model
def evaluate(test_loader, model, epoch):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for data, labels, indexes in test_loader:
        ind = indexes.cpu().numpy().transpose()
        data = Variable(data).cuda()
        labels = Variable(labels).cuda()
        logits = model(data)
        outputs = F.softmax(logits, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
    
        # For SIGUA_SL
        if args.model_type=='sigua_sl':
            loss_l_small, pure_ratio, loss_l_big = \
             loss_sigua_sl(logits, labels, drop_rate_schedule[epoch], ind, noise_or_not, args.sigua_rate)
            loss= loss_l_small - args.sigua_scale*loss_l_big
        # For SIGUA_BC
        elif args.model_type=='sigua_bc':
            if epoch > args.warm_up:
                loss= loss_sigua_bc(logits, labels, train_dataset.P, args.sigua_scale)
            else:
                loss= loss_backward(logits, labels, train_dataset.P)

    acc = 100*float(correct)/float(total)
    return acc, loss


# Predict 
def predict(test_loader, model):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    preds=[]
    true=[]
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits = model(data)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        preds.extend(pred.cpu())
        true.extend(labels)
    return preds, true


# define drop rate schedule
if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate
drop_rate_schedule = np.ones(args.n_epoch)*forget_rate
drop_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)

# Adjust learning rate and betas for Adam Optimizer
def get_alpha_beta(args):
    alpha_plan = [args.lr] * args.n_epoch
    beta1_plan = [0.9] * args.n_epoch
    beta2_plan = [0.999] * args.n_epoch
    for i in range(args.epoch_decay_start, args.n_epoch):
        alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * args.lr
        beta1_plan[i] = 0.1
        beta2_plan[i] = 0.1
    return alpha_plan, beta1_plan, beta2_plan

alpha_plan, beta1_plan, beta2_plan = get_alpha_beta(args)

def adjust_learning_rate_adam_sl(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        print('learning rate=', param_group['lr'])
        print('Adam betas=', param_group['betas'])
    
def adjust_learning_rate_adam_bc(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(0.9, beta2_plan[epoch]) # Only change beta2
        print('learning rate=', param_group['lr'])
        print('Adam betas=', param_group['betas'])
    
def adjust_learning_rate_sgd(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        print('learning rate=', param_group['lr'])


# Define learning rate scheduler
def learning_rate_scheduler(optimizer, args):
    if args.lr_scheduler == 'slr':
        return StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)
