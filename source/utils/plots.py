
from source.utils.imports import *
from source.economic import Economic
from source.model.model import Model


# from HDPS.source.utils.utils import *
sns.set(color_codes=True)

    
    
# from Parameters import *
# import seaborn as sns #visualisation
# import matplotlib.pyplot as plt #visualisation
# sns.set(color_codes=True)




def Histograms_All( data, edu, type, plots_base_dir, epoch):
    plt.figure(figsize=(15,7))
    plt.hist(data[edu > 0].view(-1), edgecolor='skyblue', bins=400, alpha= .4, label='edu = 1' )
    plt.hist(data[edu <= 0].view(-1), edgecolor='orange', bins=400, alpha= .4, label='edu = 0' )
    plt.legend()
    plt.title(f'Histogram of all {type}') 
    plt.savefig(f'{plots_base_dir}/epoch{epoch}/histograms/hist_{type}_all.png')
    plt.close()
    
    
def Histogram_Retirement_Age(p, edu,  plots_base_dir = None, epoch = None, save = False):
    age_retirement = torch.concat([p, torch.ones((len(p), 1))], dim = -1).argmax(dim = -1)+ Economic.T_ER - 1 
    plt.hist(age_retirement[edu<=0], bins = 8, range=(62,70),edgecolor='orange', alpha= .4, label='edu = 0')
    plt.hist(age_retirement[edu>0], bins= 8, range=(62,70), edgecolor='skyblue', alpha= .8, label='edu = 1')

        
    plt.legend()
    
    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}/histograms'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/hist_age_retirement.png')
        plt.close()
        return 
    
    

def Histograms_Individual_Ages(data, edu, type, plots_base_dir = None, epoch = None, save = False):
    
    AGE_0 = Economic.AGE_0
    flag = type in ['Consumption', 'Asset']
        
    fig, ax = plt.subplots(1, 3+flag, figsize=(20,5))
    fig.suptitle(f'Histogram of {type} for ages 25, 40, and 55')
    ax[0].hist(data[edu > 0][:,25 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .2, label='edu = 1')
    ax[0].hist(data[edu <= 0][:,25 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .1, label='edu = 0')
    ax[0].set_title('Age 25')
    
    
    ax[1].hist(data[edu > 0][:,40 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .2, label='edu = 1')
    ax[1].hist(data[edu <= 0][:,40 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .1, label='edu = 0')
    ax[1].set_title('Age 40')
    
    ax[2].hist(data[edu > 0][:,55 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .2, label='edu = 1')
    ax[2].hist(data[edu <= 0][:,55 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .1, label='edu = 0')
    ax[2].set_title('Age 55')
    
    
    if flag:
        ax[3].hist(data[edu > 0][:,75 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .1, label='edu = 1')
        ax[3].hist(data[edu <= 0][:,75 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .1, label='edu = 0')
        ax[3].set_title('Age 75')
        
    
    plt.legend()
    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}/histograms'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/hist_{type}_25_40_55.png')
        plt.close()
        return 
    # plt.show()
    

    
    
    
def plot_trend(data:torch.Tensor, edu, pr, type, func,plots_base_dir= None, epoch = None, save = False):
    
    AGE_0 = Economic.AGE_0
    T_ER = Economic.T_ER
    
    # print(mask_retirement.shape)
    mask_retirement = torch.concat([torch.zeros(data.shape[0], data.shape[1]-pr.shape[1]),torch.cumsum(pr, dim = 1)], dim = 1)
    print(mask_retirement.shape)
    
    plt.figure(figsize=(15,7))
    
    if type in ['Income', 'Workhour']:
    
        # mask_all = torch.concat([torch.ones(len(edu), (T_ER - AGE_0)), mask_retirement], dim = 1)
        data[mask_retirement!=0] = torch.nan
    
    # elif type == 'Consumption':
    
    #     # mask_all = torch.ones_like(data)
        
    # elif type == 'Asset':
    #     mask_all = torch.ones_like(data)

    
    if func == 'mean':
        

            # plt.plot( data[edu > 0][:,:Economic.T_LR].sum(dim=0)/mask_all[edu > 0].sum(dim = 0), color='skyblue', label='edu = 1')
            # plt.plot( data[edu <= 0][:,:Economic.T_LR].sum(dim=0)/mask_all[edu <= 0].sum(dim = 0), color='orange', label='edu = 0')
            # plt.xticks(AGE_0, Economic.T_LR+1)
            
            plt.plot( data[edu > 0].nanmean(dim = 0), color='skyblue', label='edu = 1')
            plt.plot( data[edu <= 0].nanmean(dim=0), color='orange', label='edu = 0')
            # plt.xticks(AGE_0, len(data) + AGE_0)


        
    elif func == 'median':
        

        plt.plot( data[edu > 0].nanmedian(dim=0).values, color='skyblue', label='edu = 1')
        plt.plot(data[edu <= 0].nanmedian(dim=0).values, color='orange', label='edu = 0')

            
        # elif type in ['Consumption', 'Asset']:
        #     plt.plot( range(AGE_0, len(data[0]) + AGE_0), data[edu > 0].median(dim=0).values, color='skyblue', label='edu = 1')
        #     plt.plot( range(AGE_0, len(data[0]) + AGE_0), data[edu <= 0].median(dim=0).values, color='orange', label='edu = 0')

         
    plt.legend()
    plt.title(f'Trend of {type}') 
    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}/trend'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/trend_{type}_{func}.png')
        plt.close()
        
        
# def policy_function_plot(model, edu, type, all_a, all_theta,  plots_base_dir, epoch):
    
#     AGE_0 = Economic.AGE_0
#     T_LR = Economic.T_LR
    
#     fig, ax = plt.subplots(2, 4, figsize=(20,10))
#     for i, age in enumerate([25, 40, 55, 75]):

            
#         sigma_1 = all_a[edu>0][:,age - AGE_0].std()
#         sigma_0 = all_a[edu<=0][:,age - AGE_0].std()
        
#         mean_1 = all_a[edu>0][:,age - AGE_0].mean()
#         mean_0 = all_a[edu<=0][:,age - AGE_0].mean()
        
#         t_q1_1 = torch.quantile(all_theta[edu>0][:,age - AGE_0], .25)
#         t_q1_0 = torch.quantile(all_theta[edu<=0][:,age - AGE_0], .25)
        
#         t_q2_1 = torch.quantile(all_theta[edu>0][:,age - AGE_0], .5)
#         t_q2_0 = torch.quantile(all_theta[edu<=0][:,age - AGE_0], .5)
        
#         t_q3_1 = torch.quantile(all_theta[edu>0][:,age - AGE_0], .75)
#         t_q3_0 = torch.quantile(all_theta[edu<=0][:,age - AGE_0], .75)
        
        
        
#         y = torch.ones((1000, age - AGE_0)) * 50000
        
#         generated_a_1 = torch.arange(mean_1 - 3*sigma_1, mean_1 + 3 * sigma_1, 6 * sigma_1 /1000 )
#         generated_a_0 = torch.arange(mean_0 - 3 * sigma_1, mean_0 + 3 * sigma_0, 6 * sigma_0 /1000 )
        
        
        
        
#         if age <= T_LR:
#             block = model.work_blocks[f'year_{age}']
#         else:
#             block = model.RetirementYearBlock
        
#         curve1_1 = block(t_q1_1 * torch.one((1000,1)), edu[edu>0], generated_a_1, y )
#         curve2_1 = block(t_q2_1 * torch.one((1000,1)), edu[edu>0], generated_a_1, y )
#         curve3_1 = block(t_q3_1 * torch.one((1000,1)), edu[edu>0], generated_a_1, y )
        
        
        
#         curve1_0 = block(t_q1_0 * torch.one((1000,1)), edu[edu<=0], generated_a_0, y )
#         curve2_0 = block(t_q2_0 * torch.one((1000,1)), edu[edu<=0], generated_a_0, y )
#         curve3_0 = block(t_q3_0 * torch.one((1000,1)), edu[edu<=0], generated_a_0, y )
        
        
#         ax[1][i].plot(curve1_1, color='skyblue', label='edu = 1, q=1')
#         ax[1][i].plot(curve2_1, color='orange', label='edu = 1, q=2')
#         ax[1][i].plot(curve3_1, color='pink', label='edu = 1, q=3')


#         ax[0][i].plot(curve1_0, color='skyblue', label='edu = 0, q=1')
#         ax[0][i].plot(curve2_0, color='orange', label='edu = 0, q=2')
#         ax[0][i].plot(curve3_0, color='pink', label='edu = 0, q=3')
        
    
def inverse_wage(wage, t, edu):
    
    # if wage <= w_min:
    #     return 'w_min: not inversable'
    u = Economic.mu(edu, t+Economic.AGE_0)
    theta = torch.log(wage) - u
    return theta
        
        
    
def policy_function_plot_asset(model: Model, edu, type, all_a, all_w, device, plots_base_dir= None, epoch= None, save=False): 
    
    AGE_0 = Economic.AGE_0  

    
    fig, ax = plt.subplots(2, 3, figsize=(24,10))
    for i, age in enumerate([25, 40, 55]):

            
        sigma_1 = all_a[edu>0][:,age - AGE_0].std()
        sigma_0 = all_a[edu<=0][:,age - AGE_0].std()
        
        mean_1 = all_a[edu>0][:,age - AGE_0].mean()
        mean_0 = all_a[edu<=0][:,age - AGE_0].mean()
        
        w_q1_1 = all_w[edu>0][:,age - AGE_0].quantile(.25).to(device)
        t_q1_1 = inverse_wage(w_q1_1, age, 1)
        w_q1_0 = all_w[edu<=0][:,age - AGE_0].quantile(.25).to(device)
        t_q1_0 = inverse_wage(w_q1_0, age, 0)
        
        w_q2_1 = all_w[edu>0][:,age - AGE_0].quantile(.5).to(device)
        t_q2_1 = inverse_wage(w_q2_1, age, 1)
        w_q2_0 = all_w[edu<=0][:,age - AGE_0].quantile(.5).to(device)
        t_q2_0 = inverse_wage(w_q2_0, age, 0)
        
        w_q3_1 = all_w[edu>0][:,age - AGE_0].quantile(.75).to(device)
        t_q3_1 = inverse_wage(w_q3_1, age, 1)
        w_q3_0 = all_w[edu<=0][:,age - AGE_0].quantile(.75).to(device)
        t_q3_0 = inverse_wage(w_q3_0, age, 0)
        
        
        
        
        
        y = (torch.ones((1000, age - AGE_0)) * 50000).to(device)
        
        generated_a_1 = torch.arange(mean_1 - 2*sigma_1, mean_1 + 2 * sigma_1, 4 * sigma_1 /1000 ).to(device)[:1000]
        generated_a_0 = torch.arange(mean_0 - 2 * sigma_0, mean_0 + 2 * sigma_0, 4 * sigma_0 /1000 ).to(device)[:1000]
        
        
        # if age <= T_LR:
        block = model.blocks['work_blocks'][f'year_{age}']
        # else:
        #     block = model.RetirementYearBlock
        
        
        h1_1, x1_1 = block(t_q1_1 * torch.ones((1000,1)).to(device), torch.ones((1000,1)).to(device), generated_a_1.unsqueeze(1), y )
        h2_1, x2_1 = block(t_q2_1 * torch.ones((1000,1)).to(device), torch.ones((1000,1)).to(device), generated_a_1.unsqueeze(1), y )
        h3_1, x3_1 = block(t_q3_1 * torch.ones((1000,1)).to(device), torch.ones((1000,1)).to(device), generated_a_1.unsqueeze(1), y )
        
        
        
        h1_0, x1_0 = block(t_q1_0 * torch.ones((1000,1)).to(device), torch.zeros((1000,1)).to(device), generated_a_0.unsqueeze(1), y )
        h2_0, x2_0= block(t_q2_0 * torch.ones((1000,1)).to(device),  torch.zeros((1000,1)).to(device), generated_a_0.unsqueeze(1), y )
        h3_0, x3_0 = block(t_q3_0 * torch.ones((1000,1)).to(device),  torch.zeros((1000,1)).to(device), generated_a_0.unsqueeze(1), y )
        
        
        
        
        
        if type=='workhour':
            
            curve1_1 = h1_1
            curve2_1 = h2_1 
            curve3_1 = h3_1
            
            
            curve1_0 = h1_0
            curve2_0 = h2_0
            curve3_0 = h3_0
            
            ylabel = 'work hour t'
        
                  
            
            
        elif type=='Asset':
            
            y1_1 = h1_1 * w_q1_1
            y2_1 = h2_1 * w_q2_1
            y3_1 = h3_1 * w_q3_1
            
            y1_0 = h1_0 * w_q1_0
            y2_0 = h2_0 * w_q2_0
            y3_0 = h3_0 * w_q3_0
        
        
            _, curve1_1, _ = Economic.consumption_asset_cashInHand(x1_1.squeeze(), y1_1, generated_a_1, type='working') 
            _, curve2_1, _ = Economic.consumption_asset_cashInHand(x2_1.squeeze(), y2_1, generated_a_1, type='working')
            _, curve3_1, _ = Economic.consumption_asset_cashInHand(x3_1.squeeze(), y3_1, generated_a_1, type='working') 
            
            
            
            _, curve1_0, _ = Economic.consumption_asset_cashInHand(x1_0.squeeze(), y1_0, generated_a_0, type='working') 
            _, curve2_0, _ = Economic.consumption_asset_cashInHand(x2_0.squeeze(), y2_0, generated_a_0, type='working') 
            _, curve3_0, _ = Economic.consumption_asset_cashInHand(x3_0.squeeze(), y3_0, generated_a_0, type='working') 
        
            ylabel = 'Asset t+1'

        
        
        ax[1][i].plot(generated_a_1.detach().cpu(), curve1_1.detach().cpu(), color='skyblue', label=f'edu = 1, qw1={w_q1_1:.2f}')
        ax[1][i].plot(generated_a_1.detach().cpu(), curve2_1.detach().cpu(), color='orange', label=f'edu = 1, qw2={w_q2_1:.2f}')
        ax[1][i].plot(generated_a_1.detach().cpu(), curve3_1.detach().cpu(), color='pink', label=f'edu = 1, qw3={w_q3_1:.2f}')
        ax[1][i].legend()
        ax[1][i].set_title(f'{age} edu= 1')
        ax[1][i].set_xlabel(f'Asset t')
        ax[1][i].set_ylabel(ylabel)
        

        ax[0][i].plot(generated_a_0.detach().cpu(), curve1_0.detach().cpu(), color='skyblue', label=f'edu = 0, qw1={w_q1_0:.2f}')
        ax[0][i].plot(generated_a_0.detach().cpu(), curve2_0.detach().cpu(), color='orange', label=f'edu = 0, qw2={w_q2_0:.2f}')
        ax[0][i].plot(generated_a_0.detach().cpu(), curve3_0.detach().cpu(), color='pink', label=f'edu = 0, qw3={w_q3_0:.2f}')
        ax[0][i].legend()
        ax[0][i].set_title(f'{age} edu= 0')
        ax[0][i].set_xlabel(f'Asset t')
        ax[0][i].set_ylabel(ylabel)
        
        
        if save:
            dir_save = f'{plots_base_dir}/epoch{epoch}/policy'
            pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{dir_save}/policy_function_{type}_vs_asset.png')
            plt.close()
    

    
    
    
def policy_function_plot_cashInHand(model: Model, edu, type, all_a, all_w, device, plots_base_dir= None, epoch= None, save=False):
    
    AGE_0 = Economic.AGE_0  
    
    fig, ax = plt.subplots(2, 3, figsize=(24,10))
    for i, age in enumerate([25, 40, 55]):

            
        sigma_1 = all_a[edu>0][:,age - AGE_0].std()
        sigma_0 = all_a[edu<=0][:,age - AGE_0].std()
        
        mean_1 = all_a[edu>0][:,age - AGE_0].mean()
        mean_0 = all_a[edu<=0][:,age - AGE_0].mean()
        
        w_q1_1 = all_w[edu>0][:,age - AGE_0].quantile(.25).to(device)
        t_q1_1 = inverse_wage(w_q1_1, age, 1)
        w_q1_0 = all_w[edu<=0][:,age - AGE_0].quantile(.25).to(device)
        t_q1_0 = inverse_wage(w_q1_0, age, 0)
        
        w_q2_1 = all_w[edu>0][:,age - AGE_0].quantile(.5).to(device)
        t_q2_1 = inverse_wage(w_q2_1, age, 1)
        w_q2_0 = all_w[edu<=0][:,age - AGE_0].quantile(.5).to(device)
        t_q2_0 = inverse_wage(w_q2_0, age, 0)
        
        w_q3_1 = all_w[edu>0][:,age - AGE_0].quantile(.75).to(device)
        t_q3_1 = inverse_wage(w_q3_1, age, 1)
        w_q3_0 = all_w[edu<=0][:,age - AGE_0].quantile(.75).to(device)
        t_q3_0 = inverse_wage(w_q3_0, age, 0)
        
        
        
        
        
        y = (torch.ones((1000, age - AGE_0)) * 50000).to(device)
        
        generated_a_1 = torch.arange(mean_1 - 2*sigma_1, mean_1 + 2 * sigma_1, 4 * sigma_1 /1000 ).to(device)[:1000]
        generated_a_0 = torch.arange(mean_0 - 2 * sigma_0, mean_0 + 2 * sigma_0, 4 * sigma_0 /1000 ).to(device)[:1000]
        
        
        # if age <= T_LR:'
        block = model.blocks['work_blocks'][f'year_{age}']
        # else:
        #     block = model.RetirementYearBlock
        
        
        h1_1, x1_1 = block(t_q1_1 * torch.ones((1000,1)).to(device), torch.ones((1000,1)).to(device), generated_a_1.unsqueeze(1), y )
        h2_1, x2_1 = block(t_q2_1 * torch.ones((1000,1)).to(device), torch.ones((1000,1)).to(device), generated_a_1.unsqueeze(1), y )
        h3_1, x3_1 = block(t_q3_1 * torch.ones((1000,1)).to(device), torch.ones((1000,1)).to(device), generated_a_1.unsqueeze(1), y )
        
        
        
        h1_0, x1_0 = block(t_q1_0 * torch.ones((1000,1)).to(device), torch.zeros((1000,1)).to(device), generated_a_0.unsqueeze(1), y )
        h2_0, x2_0= block(t_q2_0 * torch.ones((1000,1)).to(device),  torch.zeros((1000,1)).to(device), generated_a_0.unsqueeze(1), y )
        h3_0, x3_0 = block(t_q3_0 * torch.ones((1000,1)).to(device),  torch.zeros((1000,1)).to(device), generated_a_0.unsqueeze(1), y )
        
        
        
        
        y1_1 = h1_1 * w_q1_1
        y2_1 = h2_1 * w_q2_1
        y3_1 = h3_1 * w_q3_1
        
        y1_0 = h1_0 * w_q1_0
        y2_0 = h2_0 * w_q2_0
        y3_0 = h3_0 * w_q3_0
        
        
        
        if type=='workhour':
            
            curve1_1 = h1_1
            curve2_1 = h2_1 
            curve3_1 = h3_1
            
            
            curve1_0 = h1_0
            curve2_0 = h2_0
            curve3_0 = h3_0
            
            ylabel = 'work hour t'
        
                    
 
            
            
        elif type=='Asset':
            
            
            _, curve1_1, _ = Economic.consumption_asset_cashInHand(x1_1.squeeze(), y1_1, generated_a_1, type='working') 
            _, curve2_1, _ = Economic.consumption_asset_cashInHand(x2_1.squeeze(), y2_1, generated_a_1, type='working')
            _, curve3_1, _ = Economic.consumption_asset_cashInHand(x3_1.squeeze(), y3_1, generated_a_1, type='working') 
            
            
            
            _, curve1_0, _ = Economic.consumption_asset_cashInHand(x1_0.squeeze(), y1_0, generated_a_0, type='working') 
            _, curve2_0, _ = Economic.consumption_asset_cashInHand(x2_0.squeeze(), y2_0, generated_a_0, type='working') 
            _, curve3_0, _ = Economic.consumption_asset_cashInHand(x3_0.squeeze(), y3_0, generated_a_0, type='working') 

        
            ylabel = 'Asset t+1'

        
        
        ax[1][i].plot(generated_a_1.detach().cpu() + y1_1.detach().cpu(), curve1_1.detach().cpu(), color='skyblue', label=f'edu = 1, qw1={w_q1_1:.2f}')
        ax[1][i].plot(generated_a_1.detach().cpu() + y2_1.detach().cpu(), curve2_1.detach().cpu(), color='orange', label=f'edu = 1, qw2={w_q2_1:.2f}')
        ax[1][i].plot(generated_a_1.detach().cpu() + y3_1.detach().cpu(),curve3_1.detach().cpu(), color='pink', label=f'edu = 1, qw3={w_q3_1:.2f}')
        ax[1][i].legend()
        ax[1][i].set_title(f'{age} edu= 1')
        ax[1][i].set_xlabel(f'Asset t')
        ax[1][i].set_ylabel(ylabel)
        

        ax[0][i].plot(generated_a_0.detach().cpu() + y1_0.detach().cpu(), curve1_0.detach().cpu(), color='skyblue', label=f'edu = 0, qw1={w_q1_0:.2f}')
        ax[0][i].plot(generated_a_0.detach().cpu() + y2_0.detach().cpu(), curve2_0.detach().cpu(), color='orange', label=f'edu = 0, qw2={w_q2_0:.2f}')
        ax[0][i].plot(generated_a_0.detach().cpu() + y3_0.detach().cpu(), curve3_0.detach().cpu(), color='pink', label=f'edu = 0, qw3={w_q3_0:.2f}')
        ax[0][i].legend()
        ax[0][i].set_title(f'{age} edu= 0')
        ax[0][i].set_xlabel(f'Asset t')
        ax[0][i].set_ylabel(ylabel)
        
        
        if save:
            dir_save = f'{plots_base_dir}/epoch{epoch}/policy'
            pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{dir_save}/policy_function_{type}_vs_cash_in_hand.png')
            plt.close()
    

    
    
    
       
    
def policy_function_plot_wage(model:Model, edu, type, all_a, all_w, device,  plots_base_dir= None, epoch= None, save=False):
    
    AGE_0 = Economic.AGE_0  
    w_min = Economic.minimum_wage
    
    fig, ax = plt.subplots(2, 3, figsize=(24,10))
    for i, age in enumerate([25, 40, 55]):

            
        sigma_1 = all_w[edu>0][:,age - AGE_0].std()
        sigma_0 = all_w[edu<=0][:,age - AGE_0].std()
        
        mean_1 = all_w[edu>0][:,age - AGE_0].mean()
        mean_0 = all_w[edu<=0][:,age - AGE_0].mean()
        
        a_q1_1 = all_a[edu>0][:,age - AGE_0].quantile(.25).to(device)
        # t_q1_1 = inverse_wage(w_q1_1, age, 1)
        a_q1_0 = all_a[edu<=0][:,age - AGE_0].quantile(.25).to(device)
        # t_q1_0 = inverse_wage(w_q1_0, age, 0)
        
        a_q2_1 = all_a[edu>0][:,age - AGE_0].quantile(.5).to(device)
        # t_q2_1 = inverse_wage(w_q2_1, age, 1)
        a_q2_0 = all_a[edu<=0][:,age - AGE_0].quantile(.5).to(device)
        # t_q2_0 = inverse_wage(w_q2_0, age, 0)
        
        a_q3_1 = all_a[edu>0][:,age - AGE_0].quantile(.75).to(device)
        # t_q3_1 = inverse_wage(w_q3_1, age, 1)
        a_q3_0 = all_a[edu<=0][:,age - AGE_0].quantile(.75).to(device)
        # t_q3_0 = inverse_wage(w_q3_0, age, 0)
        
        
        
        
        
        y = (torch.ones((1000, age - AGE_0)) * 50000).to(device)
        
        
        generated_w_1 = torch.arange(mean_1 - 2*sigma_1, mean_1 + 2 * sigma_1, 4 * sigma_1 /1000 ).to(device)[:1000]
        if mean_1 - 2*sigma_1 < w_min:
            generated_w_1 = torch.arange(w_min, mean_1 + 2 * sigma_1,(mean_1 + 2 * sigma_1 - w_min) /1000 ).to(device)[:1000]
        generated_th_1 = inverse_wage(generated_w_1, age, 1)
        
        generated_w_0 = torch.arange(mean_0 - 2 * sigma_0, mean_0 + 2 * sigma_0, 4 * sigma_0 /1000 ).to(device)[:1000]
        if mean_0 - 2*sigma_0 < w_min:
            generated_w_0 = torch.arange(w_min, mean_0 + 2 * sigma_0,(mean_0 + 2 * sigma_0 - w_min) /1000 ).to(device)[:1000]
        generated_th_0 = inverse_wage(generated_w_0, age, 0)
        
        # if age <= T_LR:
        block = model.blocks['work_blocks'][f'year_{age}']
        # else:
        #     block = model.RetirementYearBlock
        
        
        h1_1, x1_1 = block(generated_th_1.to(device), torch.ones((1000,1)).to(device), a_q1_1 * torch.ones((1000,1)).to(device) , y )
        h2_1, x2_1 = block(generated_th_1.to(device), torch.ones((1000,1)).to(device), a_q2_1 * torch.ones((1000,1)).to(device) , y )
        h3_1, x3_1 = block(generated_th_1.to(device), torch.ones((1000,1)).to(device), a_q3_1 * torch.ones((1000,1)).to(device) , y )
        
        
        
        h1_0, x1_0 = block(generated_th_0.to(device),  torch.zeros((1000,1)).to(device), a_q1_0 * torch.ones((1000,1)).to(device), y )
        h2_0, x2_0 = block(generated_th_0.to(device),  torch.zeros((1000,1)).to(device), a_q2_0 * torch.ones((1000,1)).to(device), y )
        h3_0, x3_0 = block(generated_th_0.to(device),  torch.zeros((1000,1)).to(device), a_q3_0 * torch.ones((1000,1)).to(device), y )
        
        
        
        
        
        if type=='workhour':
            
            curve1_1 = h1_1
            curve2_1 = h2_1 
            curve3_1 = h3_1
            
            
            curve1_0 = h1_0
            curve2_0 = h2_0
            curve3_0 = h3_0
            
            ylabel = 'work hour t'
        
            
            
            
        elif type=='Asset':
            
            y1_1 = h1_1 * generated_w_1
            y2_1 = h2_1 * generated_w_1
            y3_1 = h3_1 * generated_w_1
            
            y1_0 = h1_0 * generated_w_0
            y2_0 = h2_0 * generated_w_0
            y3_0 = h3_0 * generated_w_0
        
        
            _, curve1_1, _ = Economic.consumption_asset_cashInHand(x1_1.squeeze(), y1_1, a_q1_1)
            _, curve2_1, _ = Economic.consumption_asset_cashInHand(x2_1.squeeze(), y2_1, a_q2_1) 
            _, curve3_1, _ = Economic.consumption_asset_cashInHand(x3_1.squeeze(), y3_1, a_q3_1)
            
            
            
            _, curve1_0, _ = Economic.consumption_asset_cashInHand(x1_0.squeeze(), y1_0, a_q1_0)
            _, curve2_0, _ = Economic.consumption_asset_cashInHand(x2_0.squeeze(), y2_0, a_q2_0)
            _, curve3_0, _ = Economic.consumption_asset_cashInHand(x3_0.squeeze(), y3_0, a_q3_0)
        
            ylabel = 'Asset t+1'
            
            
            
        

        
        
        ax[1][i].plot(generated_w_1.detach().cpu(), curve1_1.detach().cpu(), color='skyblue', label=f'edu = 1, qa1 = {a_q1_1:.2f}')
        ax[1][i].plot(generated_w_1.detach().cpu(), curve2_1.detach().cpu(), color='orange', label=f'edu = 1, qa2= {a_q2_1:.2f}')
        ax[1][i].plot(generated_w_1.detach().cpu(), curve3_1.detach().cpu(), color='pink', label=f'edu = 1, qa3= {a_q3_1:.2f}')
        ax[1][i].legend()
        ax[1][i].set_title(f'{age} edu= 1')
        ax[1][i].set_xlabel(f'Wage t')
        ax[1][i].set_ylabel(ylabel)
        

        ax[0][i].plot(generated_w_0.detach().cpu(), curve1_0.detach().cpu(), color='skyblue', label=f'edu = 0, qa1={a_q1_0:.2f}')
        ax[0][i].plot(generated_w_0.detach().cpu(), curve2_0.detach().cpu(), color='orange', label=f'edu = 0, qa2= {a_q2_0:.2f}')
        ax[0][i].plot(generated_w_0.detach().cpu(), curve3_0.detach().cpu(), color='pink', label=f'edu = 0, q3 = {a_q3_0:.2f}')
        ax[0][i].legend()
        ax[0][i].set_title(f'{age} edu= 0')
        ax[0][i].set_xlabel(f'Wage t')
        ax[0][i].set_ylabel(ylabel)
        
        
        if save:
            dir_save = f'{plots_base_dir}/epoch{epoch}/policy'
            pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
            fig.savefig(f'{dir_save}/policy_function_{type}_vs_wage.png')
            plt.close()
    
    # plt.plot(a_t_1, a_t)
    
    
    
    
  
    



def plot_pr(all_p, edu, plots_base_dir= None, epoch= None, save=False):
    
    plt.title('trend asset for model')

    plt.plot(  all_p[edu>0].mean(dim=0), label='edu=1')
    plt.plot(  all_p[edu<=0].mean(dim=0), label='edu=0')
    plt.title('model')

    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/pr_mean.png')
        plt.close()
    # plt.legend()
    # plt.show()