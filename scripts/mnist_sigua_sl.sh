#!/bin/bash
#$ -S /bin/bash
#$ -cwd
for seed in 1 2 3 4 5 6 7 8 9 10
do
python main.py --dataset mnist --noise_type pairflip --noise_rate 0.45 --model_type sigua_sl --seed ${seed} --result_dir results/trial${seed} --optim adam --lr 0.001 --sigua_scale 0.001 --sigua_rate 0.7 

python main.py --dataset mnist --noise_type symmetric --noise_rate 0.5 --model_type sigua_sl --seed ${seed} --result_dir results/trial${seed} --optim adam --lr 0.001 --sigua_scale 0.01 --sigua_rate 0.3 

python main.py --dataset mnist --noise_type symmetric --noise_rate 0.2 --model_type sigua_sl --seed ${seed} --result_dir results/trial${seed} --optim adam --lr 0.001 --sigua_scale 0.01 --sigua_rate 0.4 

done
