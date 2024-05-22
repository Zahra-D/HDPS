
from imports import *
from Parameters import *

from utils import *
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
    
    

def Histograms_Individual_Ages(data, edu, type, plots_base_dir = None, epoch = None, save = False):
    
    
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
    

    
    
    
def plot_trend(data, edu, mask_retirement, type, func,plots_base_dir= None, epoch = None, save = False):
    
    plt.figure(figsize=(15,7))
    
    
    # if type in ['Income', 'Work_Hour']:
    #     weights_edu_0 = torch.concat( [torch.ones((T_ER - AGE_0)) * len(data[edu<=0]) , mask_retirement[edu<=0].sum(dim=0)], dim = 0) 
    #     weights_edu_1 = torch.concat( [torch.ones((T_ER - AGE_0)) * len(data[edu>0]) , mask_retirement[edu>0].sum(dim=0)], dim = 0) 
        
    #     plt.plot( data[edu > 0][:,:T_LR].sum(dim=0)/weights_edu_1, color='skyblue', label='edu = 1')
    #     plt.plot( data[edu <= 0][:,:T_LR].sum(dim=0)/weights_edu_0, color='orange', label='edu = 0')
        
    #     plt.xticks(AGE_0, T_LR+1)

    # elif type in ['Consumption', 'Asset']:
    if func == 'median':
        plt.plot( range(AGE_0, len(data[0]) + AGE_0), data[edu > 0].median(dim=0).values, color='skyblue', label='edu = 1')
        plt.plot( range(AGE_0, len(data[0]) + AGE_0), data[edu <= 0].median(dim=0).values, color='orange', label='edu = 0')
            # plt.xticks())
    elif func == 'mean':
        plt.plot( range(AGE_0, len(data[0]) + AGE_0), data[edu > 0].mean(dim=0), color='skyblue', label='edu = 1')
        plt.plot( range(AGE_0, len(data[0]) + AGE_0), data[edu <= 0].mean(dim=0), color='orange', label='edu = 0')
    
    plt.legend()
    plt.title(f'Trend of {type}') 
    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}/trend'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/trend_{type}_{func}.png')
        plt.close()
        
        
def policy_function_plot(model, edu, type, all_a, all_theta,  plots_base_dir, epoch):
    
    
    fig, ax = plt.subplots(2, 4, figsize=(20,10))
    for i, age in enumerate([25, 40, 55, 75]):

            
        sigma_1 = all_a[edu>0][:,age - AGE_0].std()
        sigma_0 = all_a[edu<=0][:,age - AGE_0].std()
        
        mean_1 = all_a[edu>0][:,age - AGE_0].mean()
        mean_0 = all_a[edu<=0][:,age - AGE_0].mean()
        
        t_q1_1 = torch.quantile(all_theta[edu>0][:,age - AGE_0], .25)
        t_q1_0 = torch.quantile(all_theta[edu<=0][:,age - AGE_0], .25)
        
        t_q2_1 = torch.quantile(all_theta[edu>0][:,age - AGE_0], .5)
        t_q2_0 = torch.quantile(all_theta[edu<=0][:,age - AGE_0], .5)
        
        t_q3_1 = torch.quantile(all_theta[edu>0][:,age - AGE_0], .75)
        t_q3_0 = torch.quantile(all_theta[edu<=0][:,age - AGE_0], .75)
        
        
        
        y = torch.ones((1000, age - AGE_0)) * 50000
        
        generated_a_1 = torch.arange(mean_1 - 3*sigma_1, mean_1 + 3 * sigma_1, 6 * sigma_1 /1000 )
        generated_a_0 = torch.arange(mean_0 - 3 * sigma_1, mean_0 + 3 * sigma_0, 6 * sigma_0 /1000 )
        
        
        
        
        if age <= T_LR:
            block = model.work_blocks[f'year_{age}']
        else:
            block = model.RetirementYearBlock
        
        curve1_1 = block(t_q1_1 * torch.one((1000,1)), edu[edu>0], generated_a_1, y )
        curve2_1 = block(t_q2_1 * torch.one((1000,1)), edu[edu>0], generated_a_1, y )
        curve3_1 = block(t_q3_1 * torch.one((1000,1)), edu[edu>0], generated_a_1, y )
        
        
        
        curve1_0 = block(t_q1_0 * torch.one((1000,1)), edu[edu<=0], generated_a_0, y )
        curve2_0 = block(t_q2_0 * torch.one((1000,1)), edu[edu<=0], generated_a_0, y )
        curve3_0 = block(t_q3_0 * torch.one((1000,1)), edu[edu<=0], generated_a_0, y )
        
        
        ax[1][i].plot(curve1_1, color='skyblue', label='edu = 1, q=1')
        ax[1][i].plot(curve2_1, color='orange', label='edu = 1, q=2')
        ax[1][i].plot(curve3_1, color='pink', label='edu = 1, q=3')


        ax[0][i].plot(curve1_0, color='skyblue', label='edu = 0, q=1')
        ax[0][i].plot(curve2_0, color='orange', label='edu = 0, q=2')
        ax[0][i].plot(curve3_0, color='pink', label='edu = 0, q=3')
        
    
def inverse_wage(wage, t, edu):
    
    # if wage <= w_min:
    #     return 'w_min: not inversable'
    u = mu(edu, t)
    theta = torch.log(wage) - u
    return theta
        
        
    
def policy_function_plot_asset(model, edu, type, all_a, all_w,  plots_base_dir= None, epoch= None, save=False):
    
    device = 'cuda'
    
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
        
        
        if age <= T_LR:
            block = model.work_block[f'year_{age}']
        else:
            block = model.RetirementYearBlock
        
        
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
        
                    
        elif type=='Ratio':
            
            y1_1 = h1_1 * w_q1_1
            y2_1 = h2_1 * w_q2_1
            y3_1 = h3_1 * w_q3_1
            
            y1_0 = h1_0 * w_q1_0
            y2_0 = h2_0 * w_q2_0
            y3_0 = h3_0 * w_q3_0
        
        
            curve1_1 = ((y1_1) - social_security_tax(y1_1) + generated_a_1)
            curve2_1 = ((y2_1) - social_security_tax(y2_1) + generated_a_1) 
            curve3_1 = ((y3_1) - social_security_tax(y3_1) + generated_a_1) 
            
            
            
            curve1_0 = ((y1_0) - social_security_tax(y1_0) + generated_a_0)
            curve2_0 = ((y2_0) - social_security_tax(y2_0) + generated_a_0)
            curve3_0 = ((y3_0) - social_security_tax(y3_0) + generated_a_0)
        
            ylabel = 'Ratio'    
            
            
        elif type=='Asset':
            
            y1_1 = h1_1 * w_q1_1
            y2_1 = h2_1 * w_q2_1
            y3_1 = h3_1 * w_q3_1
            
            y1_0 = h1_0 * w_q1_0
            y2_0 = h2_0 * w_q2_0
            y3_0 = h3_0 * w_q3_0
        
        
            curve1_1 = (1.0 -x1_1.squeeze())*((y1_1) - social_security_tax(y1_1) + generated_a_1)* (1+R) 
            curve2_1 = (1.0 -x2_1.squeeze())*((y2_1) - social_security_tax(y2_1) + generated_a_1)* (1+R) 
            curve3_1 = (1.0 -x3_1.squeeze())*((y3_1) - social_security_tax(y3_1) + generated_a_1)* (1+R) 
            
            
            
            curve1_0 = (1.0 -x1_0.squeeze())*((y1_0) - social_security_tax(y1_0) + generated_a_0)* (1+R) 
            curve2_0 = (1.0 -x2_0.squeeze())*((y2_0) - social_security_tax(y2_0) + generated_a_0)* (1+R) 
            curve3_0 = (1.0 -x3_0.squeeze())*((y3_0) - social_security_tax(y3_0) + generated_a_0)* (1+R) 
        
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
    
    # plt.plot(a_t_1, a_t)
    
    
    
       
    
def policy_function_plot_wage(model, edu, type, all_a, all_w,  plots_base_dir= None, epoch= None, save=False):
    
    device = 'cuda'
    
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
        
        if age <= T_LR:
            block = model.work_block[f'year_{age}']
        else:
            block = model.RetirementYearBlock
        
        
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
        
        
            curve1_1 = (1.0 -x1_1.squeeze())*((y1_1) - social_security_tax(y1_1) + a_q1_1)* (1+R) 
            curve2_1 = (1.0 -x2_1.squeeze())*((y2_1) - social_security_tax(y2_1) + a_q2_1)* (1+R) 
            curve3_1 = (1.0 -x3_1.squeeze())*((y3_1) - social_security_tax(y3_1) + a_q3_1)* (1+R) 
            
            
            
            curve1_0 = (1.0 -x1_0.squeeze())*((y1_0) - social_security_tax(y1_0) + a_q1_0)* (1+R) 
            curve2_0 = (1.0 -x2_0.squeeze())*((y2_0) - social_security_tax(y2_0) + a_q2_0)* (1+R) 
            curve3_0 = (1.0 -x3_0.squeeze())*((y3_0) - social_security_tax(y3_0) + a_q3_0)* (1+R) 
        
            ylabel = 'Asset t+1'
            
            
            
        elif type=='Ratio':
            
            y1_1 = h1_1 * generated_w_1
            y2_1 = h2_1 * generated_w_1
            y3_1 = h3_1 * generated_w_1
            
            y1_0 = h1_0 * generated_w_0
            y2_0 = h2_0 * generated_w_0
            y3_0 = h3_0 * generated_w_0
        
        
            curve1_1 = ((y1_1) - social_security_tax(y1_1) + a_q1_1)
            curve2_1 = ((y2_1) - social_security_tax(y2_1) + a_q2_1)
            curve3_1 = ((y3_1) - social_security_tax(y3_1) + a_q3_1)
            
            
            
            curve1_0 = ((y1_0) - social_security_tax(y1_0) + a_q1_0)
            curve2_0 =((y2_0) - social_security_tax(y2_0) + a_q2_0) 
            curve3_0 = ((y3_0) - social_security_tax(y3_0) + a_q3_0)
        
            ylabel = 'Ratio'

        
        
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
    
    
    
    
  
    
    
       
    
def policy_function_plot_assetx(model, edu, type, all_a, all_w,  plots_base_dir= None, epoch= None, save=False):
    
    device = 'cuda'
    
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
        
        
        if age <= T_LR:
            block = model.work_block[f'year_{age}']
        else:
            block = model.RetirementYearBlock
        
        
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
        
        
            curve1_1 = (1.0 -x1_1.squeeze())*((y1_1) - social_security_tax(y1_1) + generated_a_1)* (1+R) 
            curve2_1 = (1.0 -x2_1.squeeze())*((y2_1) - social_security_tax(y2_1) + generated_a_1)* (1+R) 
            curve3_1 = (1.0 -x3_1.squeeze())*((y3_1) - social_security_tax(y3_1) + generated_a_1)* (1+R) 
            
            
            
            curve1_0 = (1.0 -x1_0.squeeze())*((y1_0) - social_security_tax(y1_0) + generated_a_0)* (1+R) 
            curve2_0 = (1.0 -x2_0.squeeze())*((y2_0) - social_security_tax(y2_0) + generated_a_0)* (1+R) 
            curve3_0 = (1.0 -x3_0.squeeze())*((y3_0) - social_security_tax(y3_0) + generated_a_0)* (1+R) 
        
            ylabel = 'Asset t+1'

        
        
        ax[1][i].plot(generated_a_1.detach().cpu(), curve1_1.detach().cpu(), color='skyblue', label='edu = 1, q=1')
        ax[1][i].plot(generated_a_1.detach().cpu(), curve2_1.detach().cpu(), color='orange', label='edu = 1, q=2')
        ax[1][i].plot(generated_a_1.detach().cpu(), curve3_1.detach().cpu(), color='pink', label='edu = 1, q=3')
        ax[1][i].legend()
        ax[1][i].set_title(f'{age} edu= 1')
        ax[1][i].set_xlabel(f'Asset t')
        ax[1][i].set_ylabel(ylabel)
        

        ax[0][i].plot(generated_a_0.detach().cpu(), curve1_0.detach().cpu(), color='skyblue', label='edu = 0, q=1')
        ax[0][i].plot(generated_a_0.detach().cpu(), curve2_0.detach().cpu(), color='orange', label='edu = 0, q=2')
        ax[0][i].plot(generated_a_0.detach().cpu(), curve3_0.detach().cpu(), color='pink', label='edu = 0, q=3')
        ax[0][i].legend()
        ax[0][i].set_title(f'{age} edu= 0')
        ax[0][i].set_xlabel(f'Asset t')
        ax[0][i].set_ylabel(ylabel)
        
        
        if save:
            dir_save = f'{plots_base_dir}/epoch{epoch}/policy'
            pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{dir_save}/policy_function_{type}_vs_asset.png')
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