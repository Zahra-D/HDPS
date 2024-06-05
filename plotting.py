from Parameters import *
from functions import *
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from plots import *

prob = torch.tensor([P_EDU] * J)
ep_t_e = e((J, T_LR-AGE_0))
theta_t_e = torch.cumsum(ep_t_e, dim=-1) + THETA_0
edu_e = torch.bernoulli(prob)
u_t_e = mu(edu_e.unsqueeze(1), torch.arange(1, T_LR-AGE_0 + 1))
w_t_e = wage(u_t_e, theta_t_e)
dataset_eval = TensorDataset(theta_t_e, w_t_e, edu_e)
dataloader_eval = DataLoader(dataset_eval, batch_size=100000)


def do_eval_save(model, dataloader, device):
  
    
    
    all_w = []
    all_a = []
    all_h = []
    all_y = []
    all_c = []
    all_p = []
    all_t = []
    all_edu = []
    all_theta = []
    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():

            theta_t, w_t, edu = batch
            
            w_t = w_t.to(device)
            len_batch = len(batch[0])
            a_1 = torch.tensor([A_1]* len_batch)
            
            a_t, c_t_e, all_c_ER, pr, all_pr, h_t, y_t  = model(theta_t.to(device), edu.to(device), a_1.to(device),w_t)
            all_c_r = (1-pr) * ( all_pr * all_c_ER[:,:,1] + (1-all_pr) * all_c_ER[:,:,0]) +  pr * all_c_ER[:,:,2]
            
            c_t = torch.concat([c_t_e[:, :T_ER - AGE_0], all_c_r,c_t_e[:, T_LR - AGE_0+1:] ], dim = -1)
        

            all_w.extend(w_t.cpu())
            all_edu.extend(edu.cpu())
            
            all_theta.extend(theta_t.cpu())
            
            all_a.extend(a_t.cpu())
            all_h.extend(h_t.cpu())
            all_y.extend(y_t.cpu())
            all_c.extend(c_t.cpu())
            all_p.extend(pr.cpu())
            # all_t.extend(r_t.cpu())
            
            
    all_w = torch.stack(all_w)
    all_a = torch.stack(all_a)
    all_h = torch.stack(all_h)
    all_y = torch.stack(all_y)  
    all_c = torch.stack(all_c)
    all_theta = torch.stack(all_theta)
    all_p =  torch.stack(all_p)
    all_edu = torch.stack(all_edu)
    return   all_a, all_h, all_w, all_c, all_y, all_p, all_edu, all_theta
    
    # utility_retirement(all_c, all_h, epoch, s_writer,args, mode='eval')
    # draw_all_plots(base_dir, all_a, all_h, all_w, all_theta, all_c, all_y)
    
    
from model import Model
device = 'cuda'
epoch = 200

phis = [ 0.0005,0.0, 1.0]

hyperparameter_sets = []
for phi in phis:

        base_dir  = f'./Experiments/multiple_r--with_tax--gumbel--phi_{phi}/base_model_with_regu_wb_each10_10HiddenUnits_seed92_phi{float(phi)}/10000_batch_size/PSI0.04/lambda0.01/AdamW_lr:0.001'
        model = torch.load(f'{base_dir}/model/epoch{epoch}/model.pt')
        model.to(device)
        
        all_a, all_h, all_w, all_c, all_y, all_p, all_edu, all_theta = do_eval_save(model, dataloader_eval, device)
        
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
                
        policy_function_plot_asset(model, all_edu, 'workhour', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_asset(model, all_edu, 'Asset', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_asset(model, all_edu, 'Ratio', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        
        policy_function_plot_wage(model, all_edu, 'workhour', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_wage(model, all_edu, 'Asset', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)
        policy_function_plot_wage(model, all_edu, 'Ratio', all_a, all_w,  plots_base_dir = f'{base_dir}/plot', epoch = epoch, save = True)



