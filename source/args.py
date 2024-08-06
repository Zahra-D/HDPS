import argparse


def get_parser():


    parser = argparse.ArgumentParser(description='Your script description')
        

    parser.add_argument('--experiment_title', type=str)
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=92,  help='random seed for reproducibility')
    parser.add_argument('--num_hidden_unit_w', default=10, type=int, help='number of hidden units for the working network')
    parser.add_argument('--num_hidden_unit_r', default=5, type=int, help='number of hidden units for the retirement network')
    parser.add_argument('--num_epochs', type=int, default=1000,  help='number of epochs')
    parser.add_argument('--reg_mode', type=str, choices=['each10', 'last_year', 'two_years'],default='each10', help='Choose a regularization mode (each10,last_year)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-lr_phi',  type=float, default=1e-6)
    parser.add_argument('-lr_psi',  type=float, default=1e-5)
    parser.add_argument('--lmbd', type=float, default=1e-2, help='retirement lambda')
    parser.add_argument('--psi', type=float, default=0.04, help='work hour disutility coefficient')
    parser.add_argument('--save_interval_epoch', type=int, default=100, help='Number of epochs between saving model checkpoints during training')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_no', type=int, choices=[0,1],default=0, help='the index of cuda device to run the code on it')
    parser.add_argument('--save_dir', type=str,default='./Experiments', help='the directory that the training result will be saved in')
    parser.add_argument('--phi', type=float, default=0.0006, help='Phi in utility')
    parser.add_argument('--alpha_pr', type=float, default=5, help='the tempreture of gumbel-softmax funciotn of pr')
    parser.add_argument('--alpha_h', type=float, default=5, help='the tempreture of gumbel-softmax funciotn of H')
    parser.add_argument('--hard_gumbel', action='store_true', default=False, help='in gumbel softmax')

    return parser