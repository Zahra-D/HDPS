

from .imports import *
from source.economic import Economic
from source.model.model import Model
import math





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






# def get_working_block(model, year):
#   if year >= 62:
#     return model.work_retirement_blocks[f'year_{year}'].working_block
#   return model.work_blocks[f'year_{year}']
    

def cal_regu_term_each10(model: Model):

  general_regu = 0
  retier_year = 0


  for year in range(Economic.AGE_0, Economic.T_LR):
    
    if (((year-1)%10) == 0) or (year == 69):
      continue
    pin_year = (((year-1)//10)+1)*10 +1
    if year >= 62:
      # retier_year = 1
      pin_year = 69
    


    
    
    
    
    w_g_1_pin_year = model.get_working_block(pin_year).general_layer_1.weight
    # w_g_2_pin_year = get_working_block(model,pin_year).general_layer_2.weight
    
    b_g_1_pin_year = model.get_working_block(pin_year).general_layer_1.bias

    

    w_g_1_t = model.get_working_block(year).general_layer_1.weight

    b_g_1_t =  model.get_working_block(year).general_layer_1.bias

    general_regu += (torch.norm(w_g_1_pin_year[:,:year- Economic.AGE_0+3+retier_year] - w_g_1_t , 2)) / (pin_year - year)
    general_regu += (torch.norm(b_g_1_pin_year - b_g_1_t , 2)) / (pin_year - year)




  return general_regu


   

def cal_regu_term_two_before_after(model: Model):

  general_regu = 0


  for year in range(Economic.AGE_0, Economic.T_LR):
    
    
    w_g_1_t = model.get_working_block(year).general_layer_1.weight
    b_g_1_t =  model.get_working_block(year).general_layer_1.bias


    for other_year in range(max(year-2, Economic.AGE_0), min(year+2, Economic.T_LR)+1):
      if year == other_year:
        continue
      
      
      w_g_1_other_year = model.get_working_block(other_year).general_layer_1.weight
      b_g_1_other_year = model.get_working_block(other_year).general_layer_1.bias
      # print(w_g_1_other_year.shape)
      
      num_input = min(w_g_1_other_year.shape[1],w_g_1_t.shape[1] )

      general_regu += (torch.norm(w_g_1_other_year[:,:num_input] - w_g_1_t[:, :num_input] , 2)) / abs(other_year - year)
      general_regu += (torch.norm(b_g_1_other_year - b_g_1_t , 2)) / abs(other_year - year)




  return general_regu



# def cal_regu_term_each10_old(model):

#   general_regu = 0
#   task_regu_a = 0
#   task_regu_h = 0
#   retier_year = 0


#   for year in range(AGE_0,T_LR):
    
#     if (((year-1)%10) == 0) or (year == 69):
#       continue
#     pin_year = (((year-1)//10)+1)*10 +1
#     if year >= 62:
#       # retier_year = 1
#       pin_year = 69
    


    
    
    
    
#     w_g_1_pin_year = model.get_working_block(model,pin_year).general_layer_1.weight
#     w_g_2_pin_year = model.get_working_block(model,pin_year).general_layer_2.weight
    
#     b_g_1_pin_year = model.get_working_block(model,pin_year).general_layer_1.bias
#     b_g_2_pin_year = model.get_working_block(model,pin_year).general_layer_2.bias


#     w_h_pin_year = model.get_working_block(model,pin_year).task_h.task_layer.weight
#     w_a_pin_year = model.get_working_block(model,pin_year).task_a_w.task_layer.weight
    
#     b_h_pin_year = model.get_working_block(model,pin_year).task_h.task_layer.bias
#     b_a_pin_year = model.get_working_block(model,pin_year).task_a_w.task_layer.bias
    
    
    

#     w_g_1_t =  model.get_working_block(model,year).general_layer_1.weight
#     w_g_2_t =  model.get_working_block(model,year).general_layer_2.weight
#     b_g_1_t =  model.get_working_block(model,year).general_layer_1.bias
#     b_g_2_t =  model.get_working_block(model,year).general_layer_2.bias
#     general_regu += (torch.norm(w_g_1_pin_year[:,:year-AGE_0+3+retier_year] - w_g_1_t , 2) + torch.norm(w_g_2_pin_year - w_g_2_t , 2)) / (pin_year - year)
#     general_regu += (torch.norm(b_g_1_pin_year - b_g_1_t , 2) + torch.norm(b_g_2_pin_year - b_g_2_t , 2)) / (pin_year - year)



#     w_h_t =  model.get_working_block(model,year).task_h.task_layer.weight
#     b_h_t =  model.get_working_block(model,year).task_a_w.task_layer.bias
    
#     task_regu_h += torch.norm(w_h_pin_year - w_h_t , 2) / (pin_year - year)
#     task_regu_h += torch.norm(b_h_pin_year - b_h_t , 2) / (pin_year - year)


#     w_a_t =  model.get_working_block(model,year).task_h.task_layer.weight
#     b_a_t =  model.get_working_block(model,year).task_a_w.task_layer.bias
#     task_regu_a += torch.norm(w_a_pin_year - w_a_t , 2) / (pin_year - year)
#     task_regu_a += torch.norm(b_a_pin_year - b_a_t , 2) / (pin_year - year)

#   return general_regu, task_regu_a, task_regu_h






  
def total_utility(c_t, c_t_ER, pr_bar, pr_t, h_t, phi, psi, epoch, s_writer, args, mode = 'train' ):

    
  device = c_t.device
  BETA_t = torch.pow(Economic.BETA, torch.arange(Economic.T_D - Economic.AGE_0 + 1)).to(device)
  dummy_h = torch.zeros_like(c_t_ER[:, :,0]).to(device)
  i_ER = Economic.T_ER - Economic.AGE_0
  i_LR = Economic.T_LR - Economic.AGE_0
  
  
  
  
  utility_ER = Economic.utility(c_t[:,:i_ER], h_t[:,:i_ER],  phi, psi, BETA_t[:i_ER])
  

  utility_ww = Economic.utility(c_t_ER[:,:, 0], h_t[:,i_ER: i_LR +1],  phi, psi, BETA_t[i_ER: i_LR+1])
  utility_rw = Economic.utility(c_t_ER[:,:, 1], dummy_h,  phi, psi, BETA_t[i_ER: i_LR+1])
  utility_rr = Economic.utility(c_t_ER[:,:, 2], dummy_h,  phi, psi, BETA_t[i_ER: i_LR+1])
  utility_LR = (1-pr_bar) * ( (1-pr_t) * utility_ww + pr_t * utility_rw )  +  pr_bar * (utility_rr)
  

  utility_D = Economic.utility(c_t[:,i_LR+1:], h_t[:,i_LR+1:], phi, psi, BETA_t[i_LR+1:])
  
  
  total_utility =  utility_ER.sum(dim=-1) + utility_LR.sum(dim=-1) + utility_D.sum(dim=-1)

  return  total_utility


def loss_meta(h_t):
  
  
  avg_work_hour_55 = h_t[:, 55 - Economic.AGE_0].mean()
  prcentage_working_60 = (h_t[:,60 - Economic.AGE_0] > 0).float().mean()
  
  
  return (abs(avg_work_hour_55 - Economic.M_D1) + abs(prcentage_working_60 - Economic.M_D2))
  

  
def loss_function(model :Model, c_t, c_t_ER, pr_bar, pr_t ,h_t, epoch, s_writer, args):
  # avg_work_hour_55 = h_t[:, 55 - Economic.AGE_0].mean()
  # prcentage_working_60 = (h_t[:,60 - Economic.AGE_0] > 0).float().mean()

  util = total_utility( c_t, c_t_ER, pr_bar, pr_t, h_t,args.phi, args.psi, epoch, s_writer, args) 
  if args.reg_mode == 'each10':
      cal_regu_term = cal_regu_term_each10
  elif args.reg_mode == 'last_year':
      cal_regu_term = cal_regu_term_lastyear
  elif args.reg_mode == 'two_years':
      cal_regu_term = cal_regu_term_two_before_after
  general_regu =  cal_regu_term(model)


  
  
  l_G=  args.lmbd
  l_U = 1 - (l_G)
  l_M = 100
  reg_term = l_G * general_regu 
  loss =  -1 * l_U * util.mean()# + l_G* reg_term #+ l_M * (abs(avg_work_hour_55 - Economic.M_D1))
  # loss =  l_M * (avg_work_hour_55 - Economic.M_D1)

  # print( l_M * (abs(avg_work_hour_55 - Economic.M_D1) + abs(prcentage_working_60 - Economic.M_D2)))
  
  
  

  s_writer.add_scalar('Loss/reg_term',reg_term.detach().cpu(), epoch)
  s_writer.add_scalar('Loss/util_term',util.mean().detach().cpu(), epoch)
  # s_writer.add_scalar('Meta/avg_work_hour_55',avg_work_hour_55.detach().cpu(), epoch)
  # s_writer.add_scalar('Meta/prcentage_working_60',prcentage_working_60.detach().cpu(), epoch)
  # s_writer.add_scalar('Meta/phi',model.phi.detach().cpu(), epoch)
  # s_writer.add_scalar('Meta/psi',model.psi.detach().cpu(), epoch)
  # s_writer.add_scalar('Meta/phi_grad',model.phi.grad.detach().cpu(), epoch)
  # s_writer.add_scalar('Meta/psi_grad',model.psi.grad.detach().cpu(), epoch)
  
    
  
  # s_writer.add_scalar('Loss/r_term',loss_R.detach().cpu(), epoch)
  
  
  return loss  
  
  
  
  


  
  
def generating_dataset(number_samples, duration, theta_0, p_edu):
  
    #generating random noise for all people during all the years of wroking
    ep_t = Economic.e((number_samples, duration))
    theta_t = torch.cumsum(ep_t, dim=-1) + theta_0
    prob = torch.tensor([p_edu] * number_samples)
    edu = torch.bernoulli(prob)
    u_t = Economic.mu(edu.unsqueeze(1), torch.arange(1,duration+1)+Economic.AGE_0)
    w_t = Economic.wage(u_t, theta_t)

    
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

