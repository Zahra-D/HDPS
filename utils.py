
from imports import *

from functions import *
from Parameters import *


sns.set(color_codes=True)

def cal_regu_term_lastyear(model, T=40):
  w_g_1_T = model.blocks[f'year_{T}'].general_layer_1.weight
  w_g_2_T = model.blocks[f'year_{T}'].general_layer_2.weight
  
  b_g_1_T = model.blocks[f'year_{T}'].general_layer_1.bias
  b_g_2_T = model.blocks[f'year_{T}'].general_layer_2.bias


  w_h_T = model.blocks[f'year_{T}'].task_layer_h.weight
  w_a_T = model.blocks[f'year_{T}'].task_layer_a_w.weight
  
  b_h_T = model.blocks[f'year_{T}'].task_layer_h.bias
  b_a_T = model.blocks[f'year_{T}'].task_layer_a_w.bias


  general_regu = 0
  task_regu_a = 0
  task_regu_h = 0


  for year in range(1,T):


    num_input = year+1

    w_g_1_t = model.blocks[f'year_{year}'].general_layer_1.weight
    w_g_2_t = model.blocks[f'year_{year}'].general_layer_2.weight
    b_g_1_t = model.blocks[f'year_{year}'].general_layer_1.bias
    b_g_2_t = model.blocks[f'year_{year}'].general_layer_2.bias
    general_regu += (torch.norm(w_g_1_T[:,:year+1] - w_g_1_t , 2) + torch.norm(w_g_2_T - w_g_2_t , 2)) / (T - year)
    general_regu += (torch.norm(b_g_1_T - b_g_1_t , 2) + torch.norm(b_g_2_T - b_g_2_t , 2)) / (T - year)



    w_h_t = model.blocks[f'year_{year}'].task_layer_h.weight
    b_h_t = model.blocks[f'year_{year}'].task_layer_h.bias
    
    task_regu_h += torch.norm(w_h_T - w_h_t , 2) / (T - year)
    task_regu_h += torch.norm(b_h_T - b_h_t , 2) / (T - year)


    w_a_t = model.blocks[f'year_{year}'].task_layer_a_w.weight
    b_a_t = model.blocks[f'year_{year}'].task_layer_a_w.bias
    task_regu_a += torch.norm(w_a_T - w_a_t , 2) / (T - year)
    task_regu_a += torch.norm(b_a_T - b_a_t , 2) / (T - year)

  return general_regu, task_regu_a, task_regu_h





def get_working_block(model, year):
  if year >= 62:
    return model.work_retirement_block[f'year_{year}'].working_block
  return model.work_block[f'year_{year}']
    

def cal_regu_term_each10(model):

  general_regu = 0
  task_regu_a = 0
  task_regu_h = 0
  retier_year = 0


  for year in range(AGE_0,T_LR):
    
    if (((year-1)%10) == 0) or (year == 69):
      continue
    pin_year = (((year-1)//10)+1)*10 +1
    if year >= 62:
      retier_year = 1
      pin_year = 69
    


    
    
    
    
    w_g_1_pin_year = get_working_block(model,pin_year).general_layer_1.weight
    w_g_2_pin_year = get_working_block(model,pin_year).general_layer_2.weight
    
    b_g_1_pin_year = get_working_block(model,pin_year).general_layer_1.bias
    b_g_2_pin_year = get_working_block(model,pin_year).general_layer_2.bias


    w_h_pin_year = get_working_block(model,pin_year).task_layer_h.weight
    w_a_pin_year = get_working_block(model,pin_year).task_layer_a_w.weight
    
    b_h_pin_year = get_working_block(model,pin_year).task_layer_h.bias
    b_a_pin_year = get_working_block(model,pin_year).task_layer_a_w.bias
    
    
    

    w_g_1_t = get_working_block(model,year).general_layer_1.weight
    w_g_2_t =  get_working_block(model,year).general_layer_2.weight
    b_g_1_t =  get_working_block(model,year).general_layer_1.bias
    b_g_2_t =  get_working_block(model,year).general_layer_2.bias
    general_regu += (torch.norm(w_g_1_pin_year[:,:year-AGE_0+3+retier_year] - w_g_1_t , 2) + torch.norm(w_g_2_pin_year - w_g_2_t , 2)) / (pin_year - year)
    general_regu += (torch.norm(b_g_1_pin_year - b_g_1_t , 2) + torch.norm(b_g_2_pin_year - b_g_2_t , 2)) / (pin_year - year)



    w_h_t =  get_working_block(model,year).task_layer_h.weight
    b_h_t =  get_working_block(model,year).task_layer_h.bias
    
    task_regu_h += torch.norm(w_h_pin_year - w_h_t , 2) / (pin_year - year)
    task_regu_h += torch.norm(b_h_pin_year - b_h_t , 2) / (pin_year - year)


    w_a_t =  get_working_block(model,year).task_layer_a_w.weight
    b_a_t =  get_working_block(model,year).task_layer_a_w.bias
    task_regu_a += torch.norm(w_a_pin_year - w_a_t , 2) / (pin_year - year)
    task_regu_a += torch.norm(b_a_pin_year - b_a_t , 2) / (pin_year - year)

  return general_regu, task_regu_a, task_regu_h




def loss_function_retirement_max_consumption(model,c_t_w, c_t_r, h_t_w, h_t_r, r_t,criteria_loss_R, epoch, s_writer, args):

  util, r_labels = utility_retirement_max_consumption(c_t_w, c_t_r, h_t_w, h_t_r, epoch, s_writer, args) 
  if args.reg_mode == 'each10':
      cal_regu_term = cal_regu_term_each10
  elif args.reg_mode == 'last_year':
      cal_regu_term = cal_regu_term_lastyear
  general_regu, task_regu_a, task_regu_h =  cal_regu_term(model)


  one_hot_labels = F.one_hot(r_labels, num_classes=T_LR - T_ER + 1)
  
  loss_R = criteria_loss_R(r_t, one_hot_labels[:, :-1].float())
  
  
  l_G= l_h = l_a  = args.lmbd
  l_U = 1 - (l_G + l_a+ l_h + l_a)
  reg_term = l_G * general_regu + l_h * task_regu_h + l_a * task_regu_a 
  loss = -1 * l_U * util.mean() + reg_term + loss_R

  s_writer.add_scalar('Loss/reg_term',reg_term.detach().cpu(), epoch)
  s_writer.add_scalar('Loss/util_term',util.mean().detach().cpu(), epoch)
  s_writer.add_scalar('Loss/r_term',loss_R.detach().cpu(), epoch)
  
  
  return loss

def utility_retirement_max_consumption(c_t_w, c_t_r, h_t_w, h_t_r, epoch, s_writer, args, mode = 'train' ):
  
  consumption_utility_w  =  (c_t_w**(1-GAMMA))/(1-GAMMA)
  consumption_utility_r  =  (c_t_r**(1-GAMMA))/(1-GAMMA)

  
  # if torch.isnan(consumption_utility).any().item():
  #   print('devug')
    
    
  device = c_t_w.device
  BETA_t = torch.pow(BETA, torch.arange(T_D - AGE_0 + 1)).to(device)

  # h_t = torch.concat([h_t, torch.zeros()])
  # h_t += 2
  work_hour_disutility_w = ((h_t_w/H[-1]) ** (1+1/ETA))/(1+1/ETA)
  work_hour_disutility_r = ((h_t_r/H[-1]) ** (1+1/ETA))/(1+1/ETA)
  working_disutility_w =  (h_t_w > 0).int()
  working_disutility_r =  (h_t_r > 0).int()
  total_utility_w = (BETA_t[:T_ER-AGE_0] * (consumption_utility_w - PHI * working_disutility_w - args.psi * work_hour_disutility_w)).sum(dim=-1)
  # total_utility = (BETA_t * (consumption_utility  -  args.psi * work_hour_disutility - PHI * working_disutility)).sum(dim=-1)
  total_utility_r = (BETA_t[T_ER-AGE_0:, None] * (consumption_utility_r - PHI * working_disutility_r - args.psi * work_hour_disutility_r)).sum(dim=-2)

  utility_r_max_senario = total_utility_r.max(dim = -1)

  # s_writer.add_scalar(f'{mode}/con_util', (BETA_t * consumption_utility).sum(dim=-1).mean().detach().cpu(), epoch)
  # s_writer.add_scalar(f'{mode}/work_dis',(args.psi * BETA_t * work_hour_disutility).sum(dim=-1).mean().detach().cpu(), epoch)
  # # s_writer.add_scalar(f'{mode}/retirment',( PSI_R * BETA_t[-1] * retirement_utility).mean().detach().cpu(), epoch)
  # s_writer.add_scalar(f'{mode}/utility',total_utility.mean().detach().cpu(), epoch)
  
 

  return total_utility_w + utility_r_max_senario.values,  utility_r_max_senario.indices
  
  
def utility_retirement_pr(c_t, h_t, epoch, s_writer, args, mode = 'train' ):
  
  consumption_utility =  (c_t**(1-GAMMA))/(1-GAMMA)
  # consumption_utility_r  =  (c_t_r**(1-GAMMA))/(1-GAMMA)

  
  # if torch.isnan(consumption_utility).any().item():
  #   print('devug')
    
    
  device = c_t.device
  BETA_t = torch.pow(BETA, torch.arange(T_D - AGE_0 + 1)).to(device)

  # h_t = torch.concat([h_t, torch.zeros()])
  # h_t += 2
  work_hour_disutility = ((h_t/H[-1]) ** (1+1/ETA))/(1+1/ETA)
  # work_hour_disutility_r = ((h_t_r/H[-1]) ** (1+1/ETA))/(1+1/ETA)
  working_disutility =  (h_t > 0).int()
  total_utility = (BETA_t * (consumption_utility - args.phi * working_disutility - args.psi * work_hour_disutility)).sum(dim=-1)
  # total_utility = (BETA_t * (consumption_utility  -  args.psi * work_hour_disutility - PHI * working_disutility)).sum(dim=-1)
  # total_utility_r = (BETA_t[T_ER-AGE_0:, None] * (consumption_utility_r - PHI * working_disutility_r - args.psi * work_hour_disutility_r)).sum(dim=-2)

  # senario_prob = torch.cumprod(pr_t, dim = -2)[:, -1]
  # senario_prob = torch.concat([senario_prob, senario_prob[:,-1].repeat(1, T_D-T_LR)], dim = -2)
  # utility_pr = torch.einsum('bs,bs->b', total_utility_r, senario_prob)
  # 
  return  total_utility

  
def loss_function_retirement_pr_cross(model, c_t, h_t, epoch, s_writer, args):

  util = utility_retirement_pr( c_t, h_t, epoch, s_writer, args) 
  if args.reg_mode == 'each10':
      cal_regu_term = cal_regu_term_each10
  elif args.reg_mode == 'last_year':
      cal_regu_term = cal_regu_term_lastyear
  general_regu, task_regu_a, task_regu_h =  cal_regu_term(model)
  # loss_R = criteria_loss_R(pr_t[:,:,0], r_t)


  # one_hot_labels = F.one_hot(r_t, num_classes=T_LR - T_ER + 1)
  
  
  
  
  l_G= l_h = l_a  = args.lmbd
  l_U = 1 - (l_G + l_a+ l_h + l_a)
  reg_term = l_G * general_regu + l_h * task_regu_h + l_a * task_regu_a 
  loss = -1 * l_U * util.mean() + reg_term 

  
  

  s_writer.add_scalar('Loss/reg_term',reg_term.detach().cpu(), epoch)
  s_writer.add_scalar('Loss/util_term',util.mean().detach().cpu(), epoch)
  # s_writer.add_scalar('Loss/r_term',loss_R.detach().cpu(), epoch)
  
  
  return loss  
  
  
   
def loss_function_retirement_pr(model,pr_t, c_t_w, c_t_r, h_t_w, h_t_r, r_t,criteria_loss_R, epoch, s_writer, args):

  util = utility_retirement_pr(pr_t, c_t_w, c_t_r, h_t_w, h_t_r, epoch, s_writer, args) 
  if args.reg_mode == 'each10':
      cal_regu_term = cal_regu_term_each10
  elif args.reg_mode == 'last_year':
      cal_regu_term = cal_regu_term_lastyear
  general_regu, task_regu_a, task_regu_h =  cal_regu_term(model)



  # one_hot_labels = F.one_hot(r_t, num_classes=T_LR - T_ER + 1)
  
  
  
  
  l_G= l_h = l_a  = args.lmbd
  l_U = 1 - (l_G + l_a+ l_h + l_a)
  reg_term = l_G * general_regu + l_h * task_regu_h + l_a * task_regu_a 
  loss = -1 * l_U * util.mean() + reg_term 

  
  

  s_writer.add_scalar('Loss/reg_term',reg_term.detach().cpu(), epoch)
  s_writer.add_scalar('Loss/util_term',util.mean().detach().cpu(), epoch)
  # s_writer.add_scalar('Loss/r_term',loss_R.detach().cpu(), epoch)
  
  
  return loss  
  
  
  
def utility_function(x_t, a_t, h_t, w_t, epoch, s_writer, args, mode = 'train' ):



  consumption_ = x_t * (w_t * h_t + a_t[:,:-1]) + 1e-8
  consumption_utility  =  (consumption_**(1-GAMMA))/(1-GAMMA)
  
  if torch.isnan(consumption_utility).any().item():
    print('devug')
    
    
  device = w_t.device
  BETA_t = torch.pow(BETA, torch.arange(T+1)).to(device)

  work_hour_disutility = ((h_t/H[-1]) ** (1+1/ETA))/(1+1/ETA)
  retirement_utility = torch.log(T_R * (torch.sum(w_t * h_t, dim=-1)/T + a_t[:, -1]))
  total_utility = (BETA_t[:T] * (consumption_utility  -  args.psi * work_hour_disutility)).sum(dim=-1)  + PSI_R *  BETA_t[-1] * retirement_utility


  s_writer.add_scalar(f'{mode}/con_util', (BETA_t[:T] * consumption_utility).sum(dim=-1).mean().detach().cpu(), epoch)
  s_writer.add_scalar(f'{mode}/work_dis',(args.psi * BETA_t[:T] * work_hour_disutility).sum(dim=-1).mean().detach().cpu(), epoch)
  s_writer.add_scalar(f'{mode}/retirement',( PSI_R * BETA_t[-1] * retirement_utility).mean().detach().cpu(), epoch)
  s_writer.add_scalar(f'{mode}/utility',total_utility.mean().detach().cpu(), epoch)
  
 

  return total_utility 






def loss_function(model,x_t, a_t, h_t, w_t, epoch, s_writer, args):

  util = utility_function(x_t, a_t, h_t, w_t, epoch, s_writer, args) 
  if args.reg_mode == 'each10':
      cal_regu_term = cal_regu_term_each10
  elif args.reg_mode == 'last_year':
      cal_regu_term = cal_regu_term_lastyear
  general_regu, task_regu_a, task_regu_h =  cal_regu_term(model)


  l_G= l_h = l_a  = args.lmbd
  l_U = 1 - (l_G + l_a+ l_h + l_a)
  reg_term = l_G * general_regu + l_h * task_regu_h + l_a * task_regu_a 
  loss = -1 * l_U * util.mean() + reg_term
  
  s_writer.add_scalar('Loss/reg_term',reg_term.detach().cpu(), epoch)
  
  return loss




def draw_all_plots(base_dir, all_a, all_h, all_w, all_theta, all_c, all_y, epoch):
  
  
  
    T = all_w.shape[1]
    
    pathlib.Path(f'{base_dir}/results/epoch{epoch}').mkdir(parents=True, exist_ok=True)
    
    
    plt.plot(np.arange(T) + AGE_0, all_h[:, :T].mean(dim=0))
    plt.title("Mean of all indiviadls' work hour per year during life time")
    plt.ylabel('Hour')
    plt.xlabel('Age')
    plt.savefig(f'{base_dir}/results/epoch{epoch}/trend_work_hour.png')
    plt.close()
    
    
    
    
    plt.plot(np.arange(all_c.shape[1]) + AGE_0, all_c.mean(dim=0))
    plt.title("Mean of all indiviadls' Consumption during life time")
    plt.ylabel('Dollar')
    plt.xlabel('Age')
    plt.savefig(f'{base_dir}/results/epoch{epoch}/trend_consumption.png')
    plt.close()
    
    
    
    
    plt.plot(all_a.mean(dim=0))
    plt.title("Mean of all indiviadls' asset per year during life time")
    plt.ylabel('Dollar')
    plt.xlabel('Age')
    plt.savefig(f'{base_dir}/results/epoch{epoch}/trend_asset.png')
    plt.close()
    
    
    
    
    plt.plot(np.arange(T) + AGE_0, all_y[:,:T].mean(dim=0))
    plt.title("Mean of all indiviadls' income during life time")
    plt.ylabel('Dollar')
    plt.xlabel('Age')
    plt.savefig(f'{base_dir}/results/epoch{epoch}/trend_income.png')
    plt.close()
    
    
    
    plt.plot(np.arange(T) + AGE_0, all_w.mean(dim=0))
    plt.title("Mean of all indiviadls' wage during life time")
    plt.ylabel('Dollar per Hour ($)')
    plt.xlabel('Age')
    plt.savefig(f'{base_dir}/results/epoch{epoch}/trend_wage.png')
    plt.close()
    
    
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Histogram of work hour for ages 25, 40, and 55')
    ax[0].hist(all_h[:,25 - AGE_0].view(-1), bins=400, edgecolor='skyblue')
    ax[1].hist(all_h[:,40 - AGE_0].view(-1), bins=400, edgecolor='skyblue')
    ax[2].hist(all_h[:,55 - AGE_0].view(-1), bins=400, edgecolor='skyblue')
    plt.savefig(f'{base_dir}/results/epoch{epoch}/hist_ages_workhour.png')
    plt.close()
    
    
    
    # fig, ax = plt.subplots(1, 3, figsize=(15,5))
    # fig.suptitle('Histogram of Asset for ages 25, 40, and 55')
    # ax[0].hist(all_a[:,25 - AGE_0].view(-1), bins=400,  edgecolor='skyblue')
    # ax[1].hist(all_a[:,40 - AGE_0].view(-1), bins=400,  edgecolor='skyblue')
    # ax[2].hist(all_a[:,55 - AGE_0].view(-1), bins=400,  edgecolor='skyblue')
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/hist_ages_asset.png')
    # plt.close()
    
    
    
    
    # plt.scatter(all_a[:, 25-AGE_0].reshape(-1),all_a[:, 26-AGE_0].reshape(-1), s=.1 )
    # plt.title('a_(t+1) vs a_t for Age = 25')
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/asset26_vs_asset25.png')
    # plt.close()
    
    
    # plt.scatter(all_a[:, 40-AGE_0].reshape(-1),all_a[:, 41 -AGE_0].reshape(-1), s=.1 )
    # plt.title('a_(t+1) vs a_t for Age = 40')
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/asset41_vs_asset40.png')
    # plt.close()
    
    
    # plt.scatter(all_a[:, 55-AGE_0].reshape(-1),all_a[:, 56 -AGE_0].reshape(-1), s=.1 )
    # plt.title('a_(t+1) vs a_t for Age = 55')  
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/asset56_vs_asset55.png')
    # plt.close()
    
    
    
    
    # plt.scatter(all_theta[:, 25-AGE_0].reshape(-1),all_a[:, 26 -AGE_0].reshape(-1), s=.1 )
    # plt.title('a_(t+1) vs theta_t for Age = 25')
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/theta25_vs_asset26.png')
    # plt.close()
    
    
    # plt.scatter(all_theta[:, 40-AGE_0].reshape(-1),all_a[:, 41 -AGE_0].reshape(-1), s=.1 )
    # plt.title('a_(t+1) vs theta_t for Age = 40')
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/theta40_vs_asset41.png')
    # plt.close()
    
    
    # plt.scatter(all_theta[:, 55-AGE_0].reshape(-1),all_a[:, 56 -AGE_0].reshape(-1), s=.1 )
    # plt.title('a_(t+1) vs theta_t for Age = 55')
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/theta55_vs_asset56.png')
    # plt.close()
    
  
  
  
  
    # age = 35

    # q1 = all_a[:,age - AGE_0].quantile(.25)
    # q2 = all_a[:,age - AGE_0].quantile(.5)
    # q3 =all_a[:,age - AGE_0].quantile(.75)
    # q4 = all_a[:,age - AGE_0].quantile(1)

    # w_1 = all_w[:,age-AGE_0][all_a[:, age-AGE_0]<=q1]
    # h_1 = all_h[:,age-AGE_0][all_a[:, age-AGE_0]<=q1]

    # w_2 = all_w[:,age-AGE_0][(all_a[:, age-AGE_0]>q1) & (all_a[:, age-AGE_0]<=q2)]
    # h_2 = all_h[:,age-AGE_0][(all_a[:, age-AGE_0]>q1) & (all_a[:, age-AGE_0]<=q2)]

    # w_3 = all_w[:,age-AGE_0][(all_a[:, age-AGE_0]>q2) & (all_a[:, age-AGE_0]<=q3)]
    # h_3 = all_h[:,age-AGE_0][(all_a[:, age-AGE_0]>q2) & (all_a[:, age-AGE_0]<=q3)]

    # w_4 = all_w[:,age-AGE_0][(all_a[:, age-AGE_0]>q3) & (all_a[:, age-AGE_0]<=q4)]
    # h_4 = all_h[:,age-AGE_0][(all_a[:, age-AGE_0]>q3) & (all_a[:, age-AGE_0]<=q4)]

    # # w_120 = all_w[:,age-AGE_0][(all_w[:, age-AGE_0]>90) & (all_w[:, age-AGE_0]<=120)]
    # # h_120 = all_h[:,age-AGE_0][(all_w[:, age-AGE_0]>90) & (all_w[:, age-AGE_0]<=120)]

    # # w_150 = all_w[:,age-AGE_0][(all_w[:, age-AGE_0]>120) & (all_w[:, age-AGE_0]<=150)]
    # # h_150 = all_h[:,age-AGE_0][(all_w[:, age-AGE_0]>120) & (all_w[:, age-AGE_0]<=150)]
    # plt.scatter(w_1, h_1, s=.3, label=f'asset <= {q1//1000}K')
    # plt.scatter(w_2, h_2, s=0.3, label=f'{q1//1000}K < asset <= {q2//1000}K')
    # plt.scatter(w_3, h_3, s=0.3, label=f'{q2//1000}K < asset <= {q3//1000}K')
    # plt.scatter(w_4, h_4, s=0.3, label=f'{q3//1000}K < asset <= {q4//1000}K')

    # plt.xlabel('wage')
    # plt.ylabel('work hour')
    # plt.legend()
    # plt.savefig(f'{base_dir}/results/epoch{epoch}/asset_wage_whour.png')
    # plt.close()
    
    
  
  
  
def generating_dataset(number_samples, duration, theta_0, p_edu):
  
    ep_t = e((number_samples, duration))
    theta_t = torch.cumsum(ep_t, dim=-1) + theta_0
    prob = torch.tensor([p_edu] * number_samples)
    edu = torch.bernoulli(prob)
    u_t = mu(edu.unsqueeze(1), torch.arange(1,duration+1))
    w_t = wage(u_t, theta_t)

    
    return TensorDataset(theta_t, w_t, edu)
   
  
  



def save_checkpoint(model, optimizer, base_dir, epoch):
    
    pathlib.Path(f'{base_dir}/epoch{epoch}').mkdir(parents=True)


                
    torch.save(optimizer.state_dict(), f'{base_dir}/epoch{epoch}/optimizer_state.pth')
    rng_checkpoint = {
        'torch_rng_state':
            torch.get_rng_state(),
        'cuda_rng_state':
            torch.cuda.get_rng_state_all(),
        'numpy_rng_state':
            np.random.get_state(),
        'python_rng_state':
            random.getstate()
    }
    
    with open(f'{base_dir}/epoch{epoch}/rng_checkpoint.pkl', 'wb') as f:
        pickle.dump(rng_checkpoint, f)
    
    torch.save(model, f"{base_dir}/epoch{epoch}/model.pt")

