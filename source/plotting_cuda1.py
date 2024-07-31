
from torch.utils.data import DataLoader
from utils.utils import generating_dataset
from utils.plots import *
from train.train import evaluation

# /home/zdelbari/HDPS/source/Experiments/phi_psi_input_tau--0.1/base_model_with_regu_wb_each10_10HiddenUnits_seed92_phi0.0006/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:1e-05/model
dataset_eval = generating_dataset(Economic.J,  Economic.T_LR - Economic.AGE_0, Economic.THETA_0,Economic.P_EDU)
dataloader_eval = DataLoader(dataset_eval, batch_size=100000)

device = 'cuda:0'
epoch = 999

alphas = [1.0, 1e-1, 1e-2, 1e-3]
lrs =  [1e-5, 1e-4, 1e-3, 1e-2]


for lr in lrs:
    for alpha in alphas:
        
        # /home/zdelbari/HDPS/source/Experiments/phi_psi_input_tau--1.0/base_model_with_regu_wb_each10_10HiddenUnits_seed92_phi0.0006/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:0.01/model/epoch900
        base_dir  = f'./Experiments/phi_psi_input_tau--{alpha}/base_model_with_regu_wb_each10_10HiddenUnits_seed92_phi0.0006/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:{lr}'
        model = torch.load(f'{base_dir}/model/epoch{epoch}/model.pt')
        model.to(device)
        
        all_a, all_h, all_w, all_c, all_y, all_p, all_edu, all_theta = evaluation(model, dataloader_eval, device)
        
        plot_pr(all_p, all_edu,plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        Histograms_Individual_Ages(all_a, all_edu, 'Asset', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_c, all_edu, 'Consumption', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_h, all_edu, 'Workhour', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_y, all_edu, 'Income', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        plot_trend(all_a, all_edu, None, 'Asset',func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_c, all_edu, None, 'Consumption',func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_h, all_edu, None, 'Workhour',func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_y, all_edu, None, 'Income' ,func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_a, all_edu, None, 'Asset',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_c, all_edu, None, 'Consumption',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_h, all_edu, None, 'Workhour',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_y, all_edu, None, 'Income',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
                
        policy_function_plot_asset(model, all_edu, 'workhour', all_a, all_w,device ,plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_asset(model, all_edu, 'Asset', all_a, all_w, device,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_asset(model, all_edu, 'Ratio', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        policy_function_plot_wage(model, all_edu, 'workhour', all_a, all_w,device,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_wage(model, all_edu, 'Asset', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_wage(model, all_edu, 'Ratio', all_a, all_w,device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)



