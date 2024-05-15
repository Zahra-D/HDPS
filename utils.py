
from imports import *

from functions import *
from Parameters import *


sns.set(color_codes=True)

def cal_regu_term_lastyear(model, T=40):
  w_g_1_T = model.blocks[f'year_{T}'].general_layer_1.weight
  w_g_2_T = model.blocks[f'year_{T}'].general_layer_2.weight
  
  b_g_1_T = model.blocks[f'year_{T}'].general_layer_1.bias
  b_g_2_T = model.blocks[f'year_{T}'].general_layer_2.bias


  w_h_T = model.blocks[f'year_{T}'].task_h.task_layer.weight
  w_a_T = model.blocks[f'year_{T}'].task_a_w.task_layer.weight
  
  b_h_T = model.blocks[f'year_{T}'].task_h.task_layer.bias
  b_a_T = model.blocks[f'year_{T}'].task_a_w.task_layer.bias


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



    w_h_t = model.blocks[f'year_{year}'].task_h.task_layer.weight
    b_h_t = model.blocks[f'year_{year}'].task_h.task_layer.bias
    
    task_regu_h += torch.norm(w_h_T - w_h_t , 2) / (T - year)
    task_regu_h += torch.norm(b_h_T - b_h_t , 2) / (T - year)


    w_a_t = model.blocks[f'year_{year}'].task_layer_a_w.weight
    b_a_t = model.blocks[f'year_{year}'].task_layer_a_w.bias
    task_regu_a += torch.norm(w_a_T - w_a_t , 2) / (T - year)
    task_regu_a += torch.norm(b_a_T - b_a_t , 2) / (T - year)

  return general_regu, task_regu_a, task_regu_h






def get_working_block(model, year):
  if year >= 62:
    return model.work_retirement_blocks[f'year_{year}'].working_block
  return model.work_blocks[f'year_{year}']
    

def cal_regu_term_each10(model):

  general_regu = 0
  retier_year = 0


  for year in range(AGE_0,T_LR):
    
    if (((year-1)%10) == 0) or (year == 69):
      continue
    pin_year = (((year-1)//10)+1)*10 +1
    if year >= 62:
      # retier_year = 1
      pin_year = 69
    


    
    
    
    
    w_g_1_pin_year = get_working_block(model,pin_year).general_layer_1.weight
    # w_g_2_pin_year = get_working_block(model,pin_year).general_layer_2.weight
    
    b_g_1_pin_year = get_working_block(model,pin_year).general_layer_1.bias

    

    w_g_1_t = get_working_block(model,year).general_layer_1.weight

    b_g_1_t =  get_working_block(model,year).general_layer_1.bias

    general_regu += (torch.norm(w_g_1_pin_year[:,:year-AGE_0+3+retier_year] - w_g_1_t , 2)) / (pin_year - year)
    general_regu += (torch.norm(b_g_1_pin_year - b_g_1_t , 2)) / (pin_year - year)




  return general_regu




def cal_regu_term_each10_old(model):

  general_regu = 0
  task_regu_a = 0
  task_regu_h = 0
  retier_year = 0


  for year in range(AGE_0,T_LR):
    
    if (((year-1)%10) == 0) or (year == 69):
      continue
    pin_year = (((year-1)//10)+1)*10 +1
    if year >= 62:
      # retier_year = 1
      pin_year = 69
    


    
    
    
    
    w_g_1_pin_year = get_working_block(model,pin_year).general_layer_1.weight
    w_g_2_pin_year = get_working_block(model,pin_year).general_layer_2.weight
    
    b_g_1_pin_year = get_working_block(model,pin_year).general_layer_1.bias
    b_g_2_pin_year = get_working_block(model,pin_year).general_layer_2.bias


    w_h_pin_year = get_working_block(model,pin_year).task_h.task_layer.weight
    w_a_pin_year = get_working_block(model,pin_year).task_a_w.task_layer.weight
    
    b_h_pin_year = get_working_block(model,pin_year).task_h.task_layer.bias
    b_a_pin_year = get_working_block(model,pin_year).task_a_w.task_layer.bias
    
    
    

    w_g_1_t = get_working_block(model,year).general_layer_1.weight
    w_g_2_t =  get_working_block(model,year).general_layer_2.weight
    b_g_1_t =  get_working_block(model,year).general_layer_1.bias
    b_g_2_t =  get_working_block(model,year).general_layer_2.bias
    general_regu += (torch.norm(w_g_1_pin_year[:,:year-AGE_0+3+retier_year] - w_g_1_t , 2) + torch.norm(w_g_2_pin_year - w_g_2_t , 2)) / (pin_year - year)
    general_regu += (torch.norm(b_g_1_pin_year - b_g_1_t , 2) + torch.norm(b_g_2_pin_year - b_g_2_t , 2)) / (pin_year - year)



    w_h_t =  get_working_block(model,year).task_h.task_layer.weight
    b_h_t =  get_working_block(model,year).task_a_w.task_layer.bias
    
    task_regu_h += torch.norm(w_h_pin_year - w_h_t , 2) / (pin_year - year)
    task_regu_h += torch.norm(b_h_pin_year - b_h_t , 2) / (pin_year - year)


    w_a_t =  get_working_block(model,year).task_h.task_layer.weight
    b_a_t =  get_working_block(model,year).task_a_w.task_layer.bias
    task_regu_a += torch.norm(w_a_pin_year - w_a_t , 2) / (pin_year - year)
    task_regu_a += torch.norm(b_a_pin_year - b_a_t , 2) / (pin_year - year)

  return general_regu, task_regu_a, task_regu_h






  
def utility_retirement_pr(c_t, c_t_ER, pr_bar, pr_t, h_t, epoch, s_writer, args, mode = 'train' ):

    
  device = c_t.device
  BETA_t = torch.pow(BETA, torch.arange(T_D - AGE_0 + 1)).to(device)
  dummy_h = torch.zeros_like(c_t_ER[:, :,0]).to(device)
  i_ER = T_ER - AGE_0
  i_LR = T_LR - AGE_0
  
  utility_ER = utility(c_t[:,:i_ER], h_t[:,:i_ER], BETA_t[:i_ER], args)
  


  utility_ww = utility(c_t_ER[:,:, 0], h_t[:,i_ER: i_LR +1], BETA_t[i_ER: i_LR+1], args)
  utility_rw = utility(c_t_ER[:,:, 1], dummy_h, BETA_t[i_ER: i_LR+1], args)
  utility_rr = utility(c_t_ER[:,:, 2], dummy_h, BETA_t[i_ER: i_LR+1], args)
  utility_LR = (1-pr_bar) * ( (1-pr_t) * utility_ww + pr_t * utility_rw )  +  pr_bar * (utility_rr)
  

  utility_D = utility(c_t[:,i_LR+1:], h_t[:,i_LR+1:], BETA_t[i_LR+1:],  args)
  total_utility =  utility_ER.sum(dim=-1) + utility_LR.sum(dim=-1) + utility_D.sum(dim=-1)

  return  total_utility


def utility(c_t, h_t,BETA_t, args):
  consumption_utility =  (c_t**(1-GAMMA))/(1-GAMMA)
  work_hour_disutility = ((h_t/H[-1]) ** (1+1/ETA))/(1+1/ETA)
  working_disutility =  (h_t > 0).int()
  utility = (BETA_t * (consumption_utility - args.phi * working_disutility - args.psi * work_hour_disutility))
  return utility

  
  

  
def loss_function_retirement_pr_cross(model, c_t, c_t_ER, pr_bar, pr_t ,h_t, epoch, s_writer, args):

  util = utility_retirement_pr( c_t, c_t_ER, pr_bar, pr_t, h_t, epoch, s_writer, args) 
  if args.reg_mode == 'each10':
      cal_regu_term = cal_regu_term_each10
  elif args.reg_mode == 'last_year':
      cal_regu_term = cal_regu_term_lastyear
  general_regu =  cal_regu_term(model)
  # general_regu, task_regu_a, task_regu_h =  cal_regu_term(model)

  
  
  
  
  # l_G= l_h = l_a  = args.lmbd
  # l_U = 1 - (l_G + l_a+ l_h + l_a)
  # reg_term = l_G * general_regu + l_h * task_regu_h + l_a * task_regu_a 
  # loss = -1 * l_U * util.mean() + reg_term 
    
  l_G=  args.lmbd
  l_U = 1 - (l_G)
  reg_term = l_G * general_regu 
  loss = -1 * l_U * util.mean() + reg_term 

  
  

  s_writer.add_scalar('Loss/reg_term',reg_term.detach().cpu(), epoch)
  s_writer.add_scalar('Loss/util_term',util.mean().detach().cpu(), epoch)
  # s_writer.add_scalar('Loss/r_term',loss_R.detach().cpu(), epoch)
  
  
  return loss  
  
  
  
  


  
  
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

