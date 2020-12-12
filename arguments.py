import argparse

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--lr', type = float, default = 0.001)
   parser.add_argument('--optim', type = str, help = 'optimizer, choices: adam, sgd_mom', default = 'adam')
   parser.add_argument('--lr_scheduler', type = str, help = 'scheduler, choices: linear, slr', default = 'linear')
   parser.add_argument('--lr_decay_step', type = int, help = 'slr scheduler lr decay step size', default = 10)
   parser.add_argument('--weight_decay', type = float, help = 'weight decay', default = 0)
   parser.add_argument('--beta1', type = float, help = 'Adam beta1', default = 0.9)
   parser.add_argument('--momentum', type = float, help = 'momentum', default = 0.1)
   
   parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
   parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
   parser.add_argument('--forget_rate', type = float, help = 'forget rate, if set, typical 0.5', default =None)
   parser.add_argument('--noise_type', type=str,help='[pairflip, symmetric]', default = 'symmetric')
   parser.add_argument('--model_type', choices = ['sigua_sl', 'sigua_bc'], help='sigua_sl: SIGUA_SL, sigua_bc: SIGUA_BC') 
   parser.add_argument('--sigua_scale', type = float, help = 'scale for gradient ascent, decimals')
   parser.add_argument('--sigua_rate', type = float, help = 'thresh for seleting how many big-loss samples to sigua_sl')
   parser.add_argument('--warm_up', type = int, help = 'in which epoch to start sigua_sl', default = 0)
   parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate.')
   parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate.')
   parser.add_argument('--top_bn', action='store_true')
   parser.add_argument('--dataset', type = str, help = 'mnist, cifar10', default = 'mnist')
   parser.add_argument('--n_epoch', type=int, default=200)
   parser.add_argument('--batch_size', type=int, default=128)
   parser.add_argument('--test_batch_size', type=int, default=128)
   parser.add_argument('--seed', type=int, default=1)
   parser.add_argument('--start_epoch', type=int, default=1)
   parser.add_argument('--print_freq', type=int, default=50)
   parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
   parser.add_argument('--num_iter_per_epoch', type=int, default=400)
   parser.add_argument('--epoch_decay_start', type=int, default=80)
   parser.add_argument('--resume', action='store_true', help='continue training')
   parser.add_argument('--save_model', action='store_true')
   
   args = parser.parse_args()
   return args


