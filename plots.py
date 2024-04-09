from Parameters import *
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
sns.set(color_codes=True)




def Histograms_All( data, edu, type, plots_base_dir, epoch):
    plt.figure(figsize=(15,7))
    plt.hist(data[edu > 0].view(-1), edgecolor='skyblue', bins=400, alpha= .4, label='edu = 1' )
    plt.hist(data[edu <= 0].view(-1), edgecolor='orange', bins=400, alpha= .4, label='edu = 0' )
    plt.legend()
    plt.title(f'Histogram of all {type}') 
    plt.savefig(f'{plots_base_dir}/epoch{epoch}/histograms/hist_{type}_all.png')
    plt.close()
    
    

def Histograms_Individual_Ages(data, edu, type, plots_base_dir, epoch):
    
    
    flag = type in ['Consumption', 'Asset']
        
    fig, ax = plt.subplots(1, 3+flag, figsize=(15,5))
    fig.suptitle('Histogram of Work Hour for ages 25, 40, and 55')
    ax[0].hist(data[edu > 0][:,25 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .4, label='edu = 1')
    ax[0].hist(data[edu <= 0][:,25 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .4, label='edu = 0')
    ax[0].set_title('Age 25')
    
    
    ax[1].hist(data[edu > 0][:,40 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .4, label='edu = 1')
    ax[1].hist(data[edu <= 0][:,40 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .4, label='edu = 0')
    ax[1].set_title('Age 40')
    
    ax[2].hist(data[edu <= 0][:,55 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .4, label='edu = 1')
    ax[2].hist(data[edu <= 0][:,55 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .4, label='edu = 0')
    ax[2].set_title('Age 55')
    
    
    if flag:
        ax[3].hist(data[edu <= 0][:,75 - AGE_0].view(-1), bins=400,  edgecolor='skyblue', alpha= .4, label='edu = 1')
        ax[3].hist(data[edu <= 0][:,75 - AGE_0].view(-1), bins=400,  edgecolor='orange', alpha= .4, label='edu = 0')
        ax[3].set_title('Age 75')
        
    
    plt.legend()
    plt.savefig(f'{plots_base_dir}/epoch{epoch}/histograms/hist_{type}_25_40_55.png')
    plt.close()
    
    
    
    
    
    
def plot_trend(data, edu, mask_retirement, type, plots_base_dir, epoch):
    
    plt.figure(figsize=(15,7))
    
    
    if type in ['Income', 'Work_Hour']:
        weights_edu_0 = torch.concat( [torch.ones((T_ER - AGE_0)) * len(data[edu<=0]) , mask_retirement[edu<=0].sum(dim=0)], dim = 0) 
        weights_edu_1 = torch.concat( [torch.ones((T_ER - AGE_0)) * len(data[edu>0]) , mask_retirement[edu>0].sum(dim=0)], dim = 0) 
        
        plt.plot( data[edu > 0][:,:T_LR].sum(dim=0)/weights_edu_1, color='skyblue', label='edu = 1')
        plt.plot( data[edu <= 0][:,:T_LR].sum(dim=0)/weights_edu_0, color='orange', label='edu = 0')
        
        plt.xticks(AGE_0, T_LR+1)

    elif type in ['Consumption', 'Asset']:
        plt.plot( data[edu > 0].median(dim=0), color='skyblue', label='edu = 1')
        plt.plot( data[edu <= 0].median(dim=0), color='orange', label='edu = 0')
        plt.xticks(AGE_0, AGE_0 + len(data[0])+1)
    
    plt.legend()
    plt.title(f'Trend of {type}') 
    plt.savefig(f'{plots_base_dir}/epoch{epoch}/trend/trend_{type}.png')
    plt.close()
    
    
        
        
def policy_function_plot(model, edu, type, all_a, all_theta,  plots_base_dir, epoch):
    
    
    fig, ax = plt.subplots(2, 4, figsize=(20,10))
    for i, age in enumerate([25, 40, 55, 75]):
        if age <= T_LR:
            block = model.blocks_wr[f'year_{age}']
        else:
            block = model.blocks_r[f'year_{age}']
            
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
    
    
    # plt.plot(a_t_1, a_t)
    