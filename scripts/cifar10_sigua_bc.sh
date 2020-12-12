#!/bin/bash

for seed in 1 2 3 4 5 6 7 8 9 10
do
python main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.45 --model_type sigua_bc --seed ${seed} --result_dir results/trial${seed} --sigua_scale 1.0 --optim sgd_mom --lr 0.001 --momentum 0.9 --beta1 0.9 --lr_decay_step 20 --lr_scheduler slr --epoch_decay_start 0

python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5 --model_type sigua_bc --seed ${seed} --result_dir results/trial${seed} --sigua_scale 1.0 --optim sgd_mom --lr 0.001 --momentum 0.9 --beta1 0.9 --lr_decay_step 20 --lr_scheduler slr --epoch_decay_start 0 --warm_up 18

python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2 --model_type sigua_bc --seed ${seed} --result_dir results/trial${seed} --sigua_scale 1.0 --optim sgd_mom --lr 0.001 --momentum 0.9 --beta1 0.9 --lr_decay_step 20 --lr_scheduler slr --epoch_decay_start 0

done

