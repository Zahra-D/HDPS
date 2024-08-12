
from source.utils.imports import *
from source.economic import Economic
# from decorator import dimension_corrector
from .retirement_block import RetirementYearBlock
from .working_block import WorkYearBlock
from .earlyRetirement_block import EarlyRetiermentBlock





    


    
    
    
        
        
        


class Model(nn.Module):
  


  def __init__(self,
               num_hidden_node_w = 10,
               num_hidden_node_r = 5,
               alpha_h = 1,
               alpha_pr = 1,
               hard_gumbel = True ):
    """
    Constructor for the Model class.

    Args:
        N (int): Number of Year blocks in the model. Number of Year Blocks that have retiremnet decision
        N_R (int): 
    """

    super().__init__()
    
    
    
    layers_dict = self.initializing_layer_dict(num_hidden_node_w)

    
    self.blocks = nn.ModuleDict({
      'work_blocks': nn.ModuleDict({ f'year_{i}': WorkYearBlock(i - Economic.AGE_0+3,
                                                                num_hidden_node = num_hidden_node_w,
                                                                alpha_h = alpha_h,
                                                                layers_dict=layers_dict) for i in range(Economic.AGE_0,Economic.T_ER) }),
      
      'work_retirement_blocks' : nn.ModuleDict({ f'year_{i}': EarlyRetiermentBlock(year=i, num_hidden_node_r=num_hidden_node_r,
                                                                                   num_hidden_node_w= num_hidden_node_w,
                                                                                   alpha_h = alpha_h,
                                                                                   alpha_pr=alpha_pr,
                                                                                   hard_gumbel=hard_gumbel,
                                                                                   layers_dict = layers_dict) for i in range(Economic.T_ER, Economic.T_LR+1) }),
      
      'retirement_blocks' : nn.ModuleDict({ f'year_{i}': RetirementYearBlock(year=i, num_hidden_unit=num_hidden_node_r) for i in range(Economic.T_LR+1, Economic.T_D+1) })})

    # self.phi = nn.Parameter(torch.tensor(phi_init))
    # self.psi = nn.Parameter(torch.tensor(psi_init))
    
    
    





  def initializing_layer_dict(self, num_hidden_node_w):
    layers_dict= {}
    layers_dict['general2'] = nn.Linear(num_hidden_node_w,num_hidden_node_w)
    torch.nn.init.xavier_uniform_(layers_dict['general2'].weight)
    
    layers_dict['task_h'] = nn.Linear(num_hidden_node_w,num_hidden_node_w, bias=True)
    torch.nn.init.xavier_uniform_(layers_dict['task_h'].weight)
    
    layers_dict['task_aw'] = nn.Linear(num_hidden_node_w,num_hidden_node_w, bias=True)
    torch.nn.init.xavier_uniform_(layers_dict['task_aw'].weight)
    
    layers_dict['task_ar'] = nn.Linear(num_hidden_node_w,num_hidden_node_w, bias=True)
    torch.nn.init.xavier_uniform_(layers_dict['task_ar'].weight)
    layers_dict['task_pr'] = nn.Linear(num_hidden_node_w,num_hidden_node_w, bias=True)
    torch.nn.init.xavier_uniform_(layers_dict['task_pr'].weight)
    
    return layers_dict
    
    
    
  def get_working_block(self, year):
      if year >= 62:
        return self.blocks['work_retirement_blocks'][f'year_{year}'].working_block
      return self.blocks['work_blocks'][f'year_{year}']


  def forward(self, theta, edu, a_1, all_w):
    """
    Forward pass of the neural network model.

    Args:
        theta (torch.Tensor): Tensor with shape [batch_size, T] representing theta values for each Year.
        a_1 (torch.Tensor): Tensor with shape [batch_size, 1] representing asset for the first year.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing two tensors:
            - h_t: Output tensor representing generated h_t values for each Year.
            - a_t: Output tensor representing generated a_t values for each Year.
    """


    # theta = theta.unsqueeze(dim=-1)
    B = theta.shape[0]
    device = theta.device

    
    i_ER = Economic.T_ER -Economic.AGE_0 
    i_LR =  Economic.T_LR - Economic.AGE_0 
    i_D = Economic.T_D - Economic.AGE_0 +1
    

    all_a = torch.zeros(B,i_D+1 ).to(device)
    all_h = torch.zeros(B, i_D).to(device)
    all_y = torch.zeros(B, i_LR).to(device)
    all_c = torch.zeros(B, i_D).to(device)
    
    # all_x_mean =torch.zeros(i_ER).to(device)
    all_x_t =torch.zeros(B, i_ER).to(device)

    
    
    
    #using the block for the first year, predicting a_2 and h_1
    h_t, x_t= self.blocks['work_blocks'][f'year_22'](theta[:, 0], edu, a_1) 
    y_t = all_w[:,0] * h_t
    
    c_t, a_t, _ = Economic.consumption_asset_cashInHand(x_t, y_t, a_1, type = 'working')
    # a_t = (1.0-x_t)*(y_t - income_tax(y_t)- social_security_tax(y_t)+ a_1)*(1+R)
    # c_t = (x_t)*(y_t - income_tax(y_t) - social_security_tax(y_t)+ a_1)+1e-8
    
    
    
    all_a[:,0] = a_1
    all_h[:,0] = h_t
    all_y[:,0] = y_t
    all_c[:, 0] = c_t
    all_a[:,1] = a_t
    all_x_t[:,0] = x_t
    # all_x_std[0] = x_std
    
    
    # loop over years until early retirement,
    # In each year, previously generated values of 'a_t' and the corresponding theta will be fed to the model.
    for i in range(1,i_ER):
      

      h_t, x_t = self.blocks['work_blocks'][f'year_{i+Economic.AGE_0}'](theta[:, i], edu, a_t, all_y[:, :i])
      # all_x_mean[i] = x_mean
      all_x_t[:,i] = x_t
      
      y_t = all_w[:,i] * h_t
      c_t , a_t , _ = Economic.consumption_asset_cashInHand(x_t, y_t, a_t, type = 'working')
      # all_c[:, i] = (x_t)*(y_t - income_tax(y_t) -social_security_tax(y_t)+ a_t) +1e-8
      # a_t = (1.0 -x_t)*((y_t) - income_tax(y_t) -  social_security_tax(y_t) + a_t)* (1+R) 
      
      all_y[:, i] = y_t
      all_h[:, i] = h_t
      all_a[:,i+1] = a_t
      all_c[:,i] = c_t
    


    
    

    
    #the probability of becoming retired at year 61 (it is zero)
    pr_bar= torch.zeros_like(a_t)
    
    
    #benefit if retire at age 62
    b_bar = torch.zeros_like(a_t)
    
    
    #state of retirement in the begining of each year, (index 0 == year 62, index -1 == year 70)
    all_pr_bar = torch.zeros(B, i_LR - i_ER+1).to(device)
    all_pr = torch.zeros(B, i_LR - i_ER+1).to(device)
    
    #index 0 for ww
    #index 1 for rw
    #index 2 for rr
    all_c_ER = torch.zeros(B, i_LR - i_ER+1, 3).to(device)

    
    a_w_t = a_r_t = a_t
    
    for i in range(i_ER, i_LR):
      # t_t = torch.ones_like(a_t) * (i+AGE_0)
      # benefit = retirement_benefit(all_y[:, :i], i - i_ER, 35)

      # f forward(self, theta, edu, a_t, all_y, w_t, b_t, pr_bar_t, b_bar_t):

      outputs = self.blocks['work_retirement_blocks'][f'year_{i+Economic.AGE_0}'](theta[:, i], edu, a_w_t,  a_r_t, all_y[:, :i,],all_w[:, i],  pr_bar, b_bar)

      #if we are retired (pr = 1) the working hour should be consider 0 in utility calculation
      all_h[:, i] = outputs['h_t']
      all_pr_bar[:, i - i_ER+1] = outputs['pr_bar_tp']
      all_y[:, i] = outputs['y_t']
 
      all_a[:, i+1] = outputs['a_tp']
      
      all_c_ER[:, i-i_ER, 0] = outputs['c_ww_t']
      all_c_ER[:, i-i_ER, 1] = outputs['c_rw_t']
      all_c_ER[:, i-i_ER, 2] = outputs['c_rr_t']
      
  
      
      pr_bar = outputs['pr_bar_tp']
      b_bar = outputs['b_bar_tp']
      a_w_t = outputs['a_w_tp']
      a_r_t = outputs['a_r_tp']
      
      all_pr[:,i-i_ER] =  outputs['pr_t']
      
      
    
    
    
    
    
    
    #year 70
    outputs = self.blocks['work_retirement_blocks'][f'year_{i_LR+Economic.AGE_0}'](theta[:, i_LR-1], edu, a_w_t,  a_r_t, all_y[:, :i_LR,],all_w[:, i_LR-1],  pr_bar, b_bar)
    
    all_a[:, i_LR+1] = outputs['a_tp']
    
    
    all_c_ER[:, i_LR-i_ER, 0] = 1e-8
    all_c_ER[:, i_LR-i_ER, 1] = outputs['c_rw_t']
    all_c_ER[:, i_LR-i_ER, 2] = outputs['c_rr_t']
    
    a_r_t = outputs['a_r_tp']
    b_bar = outputs['b_bar_tp']
    all_pr[:,i_LR-i_ER] =  1
    
  
  
    



    for i in range(i_LR+1, i_D):
      
    
      # t = torch.ones_like(a_r_t).to(a_r_t.device) * (i+AGE_0)
      
      
      x_t = self.blocks['retirement_blocks'][f'year_{i+Economic.AGE_0}'](a_r_t, b_bar)
      
      
      c_t, a_r_t, _ = Economic.consumption_asset_cashInHand(x_t, b_bar, a_r_t, type = 'retired')
      # c_t = (x_t *(b_bar + a_r_t)) + 1e-8
      # a_r_t =  ((1.0-x_t)*(b_bar + a_r_t)*(1+R))
      
      
      all_a[:,i+1] = a_r_t
      all_c[:,i] = c_t

  
    return  all_a, all_c, all_c_ER, all_pr_bar, all_pr, all_h, all_y, all_x_t
     
      
      
      
    

    