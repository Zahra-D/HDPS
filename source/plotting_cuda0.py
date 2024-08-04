
from torch.utils.data import DataLoader
from utils.utils import generating_dataset
from utils.plots import *
from train.train import evaluation


dataset_eval = generating_dataset(Economic.J,  Economic.T_LR - Economic.AGE_0, Economic.THETA_0,Economic.P_EDU)
dataloader_eval = DataLoader(dataset_eval, batch_size=100000)



device = 'cuda:0'
epoch = 999

# phis = [ 0.0006, 0.0005, 0.0001,  0.001]
# alphas = [ 1e-2, 1e-1,1.0,  1e-3]
phis = [0.0006]
alphas = [1e0,1e-1, 1e-2, 1e-3]


for phi in phis:
    for alpha in alphas:
# /home/zdelbari/HDPS/source/Experiments/2year_reg/multiple_r--with_tax--gumbel--phi_0.0005--tau_0.01--hard_gumbel/base_model_with_regu_wb_two_years_10HiddenUnits_seed92_phi0.0005/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:0.001/model/epoch999/model.pt
        # base_dir  = f'./Experiments/multiple_r--with_tax--gumbel--phi_{phi}--tau_{alpha}--hard_gumbel/base_model_with_regu_wb_each10_10HiddenUnits_seed92_phi{float(phi)}/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:0.001'
        base_dir = f'./Experiments/2year_reg/multiple_r--with_tax--gumbel--phi_0.0006--tau_{alpha}--hard_gumbel/base_model_with_regu_wb_two_years_10HiddenUnits_seed92_phi0.0006/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:0.001'
        model = torch.load(f'{base_dir}/model/epoch{epoch}/model.pt')
        model.to(device)
        
        print("exp:", alpha, phi)
        
        all_a, all_h, all_w, all_c, all_y, all_p, all_edu, all_theta = evaluation(model, dataloader_eval, device)
        
        plot_pr(all_p, all_edu,plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        
        plot_trend(all_a, all_edu, all_p[:,1:], 'Asset',func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_c, all_edu, all_p[:,1:], 'Consumption',func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_h, all_edu, all_p[:,1:], 'Workhour',func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_y, all_edu, all_p[:,1:], 'Income' ,func='median', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_a, all_edu, all_p[:,1:], 'Asset',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_c, all_edu, all_p[:,1:], 'Consumption',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_h, all_edu, all_p[:,1:], 'Workhour',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        plot_trend(all_y, all_edu, all_p[:,1:], 'Income',func='mean', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
                
        
        
        Histogram_Retirement_Age(all_p , all_edu, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_a, all_edu, 'Asset', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_c, all_edu, 'Consumption', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_h, all_edu, 'Workhour', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        Histograms_Individual_Ages(all_y, all_edu, 'Income', plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
       
        policy_function_plot_cashInHand(model, all_edu, 'workhour', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_cashInHand(model, all_edu, 'Asset', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        policy_function_plot_asset(model, all_edu, 'workhour', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_asset(model, all_edu, 'Asset', all_a, all_w,device,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        # policy_function_plot_asset(model, all_edu, 'Ratio', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        policy_function_plot_wage(model, all_edu, 'workhour', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_wage(model, all_edu, 'Asset', all_a, all_w, device, plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        # policy_function_plot_wage(model, all_edu, 'Ratio', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)



