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

from model import CNN 
from arguments import get_args
from utils import train, evaluate, predict
from utils import accuracy, save_checkpoint, resume_checkpoint
from utils import train_dataset, test_dataset, input_channel, num_classes
from utils import adjust_learning_rate_adam_sl, adjust_learning_rate_adam_bc, \
                  adjust_learning_rate_sgd, learning_rate_scheduler

args = get_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

save_dir = os.path.join(args.result_dir, args.dataset, args.model_type)

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
    if args.save_model:
        os.system('mkdir -p %s' % model_dir)

model_str = args.dataset + '_' + args.model_type + '_' + args.noise_type + '_' + str(args.noise_rate)

txtfile = os.path.join(save_dir, model_str + ".txt")
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

if os.path.exists(txtfile):
    if args.resume:
        os.system('cp %s %s' % (txtfile, "%s.bak-%s" % (txtfile,nowTime)))
    else:
        os.system('mv %s %s' % (txtfile, txtfile + ".bak"))

def main():
    # Data Loader (Input Pipeline)
    print('===>Model Type:', args.model_type)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    clf = CNN(input_channel=input_channel, n_outputs=num_classes).cuda()
    print(clf.parameters)

    # Optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, \
                         betas=(args.beta1, args.momentum), weight_decay=args.weight_decay)
    if args.optim == 'sgd_mom':
        optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    # learning rate schedule
    if args.lr_scheduler == 'linear':
        adjust_lr_linear = True
    else:
        scheduler = learning_rate_scheduler(optimizer, args)
        adjust_lr_linear = False
    print('Adjust lr linear [True/False]:', adjust_lr_linear)
    
    best_test_acc=0
    mean_pure_ratio=0
    mean_pure_ratio1=0


    if not args.resume:
        epoch=0
        train_acc=0
        
        # evaluate models with random weights
        test_acc, test_loss = evaluate(test_loader, clf, epoch)
        print('Epoch [%d/%d] Test Accuracy on the 10000 test data: %.4f %%, Pure Ratio %.4f %%' % \
                 (epoch+1, args.n_epoch, test_acc, mean_pure_ratio))
        # save results
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': ' + str(train_acc) +' ' + str(test_acc) + ' ' \
                          + str(mean_pure_ratio) + "\n")
    else:
        args.start_epoch, best_test_acc, optimizer, clf = \
                               resume_checkpoint(optimizer, clf, model_dir, model_str)

    # training
    for epoch in range(args.start_epoch, args.n_epoch):
        # train models
        clf.train()

        # learning rate scheduler step
        if adjust_lr_linear:
            if args.model_type=='sigua_sl':
                if args.optim == 'adam':  
                    print('adjust learning rate adam sl')
                    adjust_learning_rate_adam_sl(optimizer, epoch)
                else:
                    adjust_learning_rate_sgd(optimizer, epoch)
            if args.model_type=='sigua_bc':
                if args.optim == 'adam':  
                    adjust_learning_rate_adam_bc(optimizer, epoch)
                else:
                    adjust_learning_rate_sgd(optimizer, epoch)
        else: 
            scheduler.step()

        print('Training %s...' % model_str)
        train_acc, pure_ratio_list = train(train_loader, epoch, clf, optimizer, args)
        # evaluate models
        print('Evaluating %s...' % model_str)
        test_acc, test_loss = evaluate(test_loader, clf, epoch)
        
        # save results
        if args.model_type=='sigua_sl':
            print('Epoch [%d/%d] Test Accuracy on the %s test data: %.4f %%, Pure Ratio %.4f %%' % \
                              (epoch+1, args.n_epoch, len(test_dataset), test_acc, mean_pure_ratio))
            mean_pure_ratio = sum(pure_ratio_list)/len(pure_ratio_list)
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_acc) + ' ' + str(test_acc) + ' ' \
                              + str(mean_pure_ratio) + "\n")
        else:
            print('Epoch [%d/%d] Test Accuracy on the %s test data: %.4f %%' % \
                         (epoch+1, args.n_epoch, len(test_dataset), test_acc))
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_acc) +' ' + str(test_acc) + "\n")

        # remember best prec@1 and save checkpoint
        is_best=test_acc > best_test_acc
        best_test_acc=max(test_acc, best_test_acc)
        if args.save_model:
            save_checkpoint({
            'epoch': epoch+1,
            'state_dict': clf.state_dict(),
            'best_prec1': best_test_acc,
            'optimizer': optimizer.state_dict(),
            }, is_best, model_dir, model_str)

    print('Predicting %s...' % model_str)
    preds, true=predict(test_loader, clf)
    for p, t in zip(preds, true):
        with open(txtfile.replace('.txt', '_pred.txt'), "a") as myfile:
            myfile.write(str(p) +' ' + str(t) + "\n")

if __name__=='__main__':
    main()
