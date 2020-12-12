#!/bin/bash

for seed in 1 2 3 4 5 6 7 8 9 10
do
python main.py --dataset mnist --noise_type pairflip --noise_rate 0.45 --model_type sigua_bc --warm_up 0 --seed ${seed} --result_dir results/trial${seed} --sigua_scale 1.0 --optim adam --lr 0.001 --momentum 0.1 --beta1 0.9 --lr_scheduler slr --lr_decay_step 10

python main.py --dataset mnist --noise_type symmetric --noise_rate 0.5 --model_type sigua_bc --warm_up 0 --seed ${seed} --result_dir results/trial${seed} --sigua_scale 1.0 --optim adam --lr 0.001 --momentum 0.1 --beta1 0.9 --lr_scheduler slr --lr_decay_step 10

python main.py --dataset mnist --noise_type symmetric --noise_rate 0.2 --model_type sigua_bc --warm_up 0 --seed ${seed} --result_dir results/trial${seed} --sigua_scale 1.0 --optim adam --lr 0.001 --momentum 0.1 --beta1 0.9 --lr_scheduler slr --lr_decay_step 10

done

