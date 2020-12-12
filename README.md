# SIGUA 
ICML'20: SIGUA: Forgetting May Make Learning with Noisy Labels More Robust 


========
This is the code for the paper:
[SIGUA: Forgetting May Make Learning with Noisy Labels More Robust](http://proceedings.mlr.press/v119/han20c.html)

Presented at [ICML'20]()
If you find this code useful in your research then please cite
```bash
@InProceedings{pmlr-v119-han20c,
  title = 	 {{SIGUA}: Forgetting May Make Learning with Noisy Labels More Robust},
  author =       {Han, Bo and Niu, Gang and Yu, Xingrui and Yao, Quanming and Xu, Miao and Tsang, Ivor and Sugiyama, Masashi},
  booktitle = 	 {International Conference on Machine Learning},
  pages = 	 {4006--4016},
  year = 	 {2020}
}
```

## Setup
Istall Miniconda3, and then

```bash
conda create -f environment.yml 
conda activate pytorch1.0.0_py36
```

## Running SIGUA on benchmark datasets (MNIST, CIFAR-10)
```bash
sh scripts/mnist_sigua_sl.sh
sh scripts/mnist_sigua_bc.sh
sh scripts/cifar10_sigua_sl.sh
sh scripts/cifar10_sigua_bc.sh
```

## The other reproducible version
Please check the other reproducible code of SIGUA: https://github.com/yeachan-kr/pytorch-sigua
