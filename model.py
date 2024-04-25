import torch
import torch.nn as nn
import torch.nn.functional as F 
from Parameters import *
from functions import *


def dimension_corrector(func):
  def wrapper(*args, **kwargs):
    
    new_args =[]
    # new_kwargs = {}
    
    for arg in args:
      if isinstance(arg, torch.Tensor) and arg.dim() <2:
        arg = arg.unsqueeze(-1)
      new_args.append(arg)
        
        
    for key, val in kwargs.items():
      if isinstance(val, torch.Tensor) and val.dim() <2:
        kwargs[key] = val.unsqueeze(-1)
        
    return func(*new_args, **kwargs)
  return wrapper


  

class WorkYearBlock(nn.Module):

  def __init__(self, num_input=1, num_hidden_node = 10, mode = 'working_year'):
    super().__init__()
    
    self.bn_input = nn.BatchNorm1d(num_input)
    
    #it could be either working_years or early_retirement_years
    self.mode = mode
    
    
    #initializing the layers
    #first two layers of NN that are  general layers
    self.general_layer_1 = nn.Linear(num_input, num_hidden_node)
    torch.nn.init.xavier_uniform_(self.general_layer_1.weight)


    self.general_layer_2 = nn.Linear(num_hidden_node,num_hidden_node)
    torch.nn.init.xavier_uniform_(self.general_layer_2.weight)

      
    

    # third layer which is task specific layer, one for h and one for a
    self.task_layer_h = nn.Linear(num_hidden_node,num_hidden_node, bias=True)
    torch.nn.init.xavier_uniform_(self.task_layer_h.weight)


    
    
    self.task_layer_a = nn.Linear(num_hidden_node,num_hidden_node, bias=True)
    torch.nn.init.xavier_uniform_(self.task_layer_a.weight)


    


    #forth layer that dedicated to year, again one for h and one for a
    # # h is categorical with 4 categories and for the final result we will get mean of all the four output
    self.year_layer_h = nn.Linear(num_hidden_node,4)
    torch.nn.init.xavier_uniform_(self.year_layer_h.weight)



    self.year_layer_a = nn.Linear(num_hidden_node,1)
    torch.nn.init.xavier_uniform_(self.year_layer_a.weight)



    # activation function, we can use a single ReLU wherever it is needed.
    self.activation_function = nn.GELU()
    self.a_activation = nn.Sigmoid()
    self.h = torch.tensor([0.0,1300.0,2080.0,2860.0])
    
    




  @dimension_corrector
  def forward(self, theta, edu, a, y = None):
    """
    Forward pass of the neural network.

    Args:
        theta (torch.Tensor): Tensor with shape [batch_size, 1] representing theta values.
        a (torch.Tensor): Tensor with shape [batch_size, t], t is depend on the year, represent the asset till now

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing two tensors:
            - x_h: Output tensor for the 'h' task, 4 weights, one for each defind work hour.
            - x_a: Output tensor for the 'a' task, the predict asset for this year .
    """



    
    
    if isinstance(y, torch.Tensor):
      x = torch.concat([theta, edu, a, y], dim = -1)
      
      
    else:
      x = torch.concat([theta, edu, a], dim = -1)
    
    
    # d = self.device
    device = x.device

    x = self.bn_input(x)

    x = self.general_layer_1(x)
    x = self.activation_function(x)
    x = self.general_layer_2(x)
    x = self.activation_function(x)


    x_h = self.task_layer_h(x)
    x_h = self.activation_function(x_h)

    x_x = self.task_layer_a(x)
    x_x = self.activation_function(x_x)

    x_h = self.year_layer_h(x_h)
    # x_h = (self.a_activation(x_h) * self.h).squeeze(-1)
    x_h = F.softmax(x_h, dim=-1)
    


    self.h = self.h.to(device)
    
    # one_hot[torch.arange(B).to(device), torch.argmax(x_h, dim = 1)] = self.h[torch.argmax(x_h, dim = 1)]
    x_h = torch.einsum('bh,h->b',x_h, self.h)
    # x_h = x_h * self.h

    x_x = self.year_layer_a(x_x)
    x_x = self.a_activation(x_x)

    if self.mode == 'early_retirement_year':
      # x_p = self.p_bn(p)
      # x = torch.concat([x, x_p], dim=-1)
      # x_r = self.task_layer_r(x)
      # x_r = self.activation_function(x_r)
      # x_r = self.year_layer_r(x_r)
      # x_r = self.activation_retirement_decision( 1000 * x_r)
      

      return x_h, x_x.squeeze(), x

    return x_h, x_x.squeeze()
  
  
class RetirementYearBlock(nn.Module):
    
    def __init__(self, num_hidden_unit = 5, mode='retirement_year', year=None):
        super().__init__()
        
        self.mode = mode
        
        self.year = year
        
        self.bn = nn.BatchNorm1d(2)
        
        self.layer_1 = nn.Linear(2, num_hidden_unit)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)

        
        self.layer_2 = nn.Linear(num_hidden_unit, 1)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        # torch.nn.init.xavier_uniform_(self.layer_3.bias)
        
        
        self.activation_function = nn.GELU()
        self.x_activation = nn.Sigmoid()
        
        
    @dimension_corrector
    def forward(self, a, b):
      x = torch.concat([a, b], dim = -1)
      x = self.bn(x)
      
      x_ = self.layer_1(x)
      x = self.activation_function(x_)
      
      x = self.layer_2(x)
      x = self.x_activation(x)
      
      if self.mode == 'early_retirement_year':
        return x.squeeze(), x_
      
 
      
      return x.squeeze()
        
        
       
#to do: do not get pr_bar as input????? 
class PrBlock(nn.Module):
        def __init__(self, num_input, alpha):
              super().__init__()
              
              self.layer1 = nn.Linear(num_input, 1)
              self.activation = nn.Sigmoid()
              self.alpha = alpha
              
              
        @dimension_corrector   
        def forward(self, x_r, x_w):
              x = torch.concat([x_w, x_r], dim=-1)
              x = self.layer1(x)
              x = self.activation(self.alpha * x)
              
              return x.squeeze()
  
    
class EarlyRetiermentBlock(nn.Module):
  
      def __init__(self, year=61, num_hidden_node_w = 10, num_hidden_node_r = 5, alpha_pr = 1):
        super().__init__()
        
        assert (year >= 62) and (year<=69)
        
        self.working_block = WorkYearBlock(num_input=year - AGE_0 + 3, num_hidden_node = num_hidden_node_w,  mode= 'early_retirement_year')
        self.retirement_block = RetirementYearBlock(mode= 'early_retirement_year', num_hidden_unit=num_hidden_node_r, year=year)
        self.pr_block = PrBlock(num_input = num_hidden_node_w + num_hidden_node_r, alpha=alpha_pr)
        self.year= year
        
        
        
        
        
        
        
        
      def forward(self, theta, edu, a_t, all_y, w_t, b_t, pr_bar_t, b_bar_t):
        
        h_t, x_w, xx_w = self.working_block(theta, edu, a_t, all_y)
        x_r, xx_r = self.retirement_block(a_t, b_t)
        

        x_rr, _ = self.retirement_block(a_t, b_bar_t)
          
          
        pr_t = self.pr_block(xx_r, xx_w)
        y_t = w_t * h_t 
        
      
  
        
        #calculating a_w a_r a_rr
        c_w_t = x_w * (y_t - social_security_tax(y_t) + a_t) + 1e-8
        a_w_tp = (1.0 - x_w)*((y_t) - social_security_tax(y_t) + a_t)* (1+R) 
        
        a_r_tp = ((1.0-x_r)*(b_t + a_t)*(1+R))
        c_r_t = (x_r *(b_t + a_t))+ 1e-8

        a_rr_tp = ((1.0-x_rr)*(b_bar_t + a_t)*(1+R))
        c_rr_t = (x_rr *(b_bar_t + a_t))+ 1e-8
        
        c_t = (1-pr_bar_t) * (  pr_t * c_r_t  +  (1-pr_t) * c_w_t  )      +      pr_bar_t*(c_rr_t)
        
        
        #update a_tp
        a_tp = (1-pr_bar_t) * (  pr_t * a_r_tp  +  (1-pr_t) * a_w_tp  )      +      pr_bar_t*(a_rr_tp)
        
        
        
        #update pr_bar
        pr_bar_tp  = pr_bar_t + (1-pr_bar_t) * pr_t
        
        #update b_bar
        b_bar_tp = pr_bar_t * b_bar_t + (1-pr_bar_t) * b_t
        
        # return {'a_tp':a_tp,
        #         'c_t': c_t,
        #         'y_t': y_t,
        #         'h_t': h_t,
        #         'pr_t': pr_t,
        #         'pr_bar_tp': pr_bar_tp, 
        #         'b_bar_tp': b_bar_tp}
        
        return {'a_tp':a_tp,
                'c_t':c_t,
                'a_w_tp': a_w_tp,
                'c_w_t': c_w_t,
                'a_r_tp': a_r_tp,
                'c_r_t': c_r_t,
                'a_rr_tp': a_rr_tp,
                'c_rr_t': c_rr_t,
                'y_t': y_t,
                'h_t': h_t,
                'pr_t': pr_t,
                'pr_bar_tp': pr_bar_tp, 
                'b_bar_tp': b_bar_tp}
        
        
        
        

        
        
        
        
        
        
        
        
        


class Model(nn.Module):
  


  def __init__(self, T_LR= 70, T_ER= 62, T_D = 82,   num_hidden_node_w = 10, num_hidden_node_r = 5, alpha_pr = 1):
    """
    Constructor for the Model class.

    Args:
        N (int): Number of Year blocks in the model. Number of Year Blocks that have retiremnet decision
        N_R (int): 
    """



    super().__init__()

    self.work_block = nn.ModuleDict({ f'year_{i}': WorkYearBlock(i - AGE_0+3, num_hidden_node = num_hidden_node_w) for i in range(AGE_0, T_ER) })
    self.work_retirement_block = nn.ModuleDict({ f'year_{i}': EarlyRetiermentBlock(year=i, num_hidden_node_r=num_hidden_node_r, num_hidden_node_w= num_hidden_node_w, alpha_pr=alpha_pr ) for i in range(T_ER, T_LR) })
    self.retirement_block = nn.ModuleDict({ f'year_{i}': RetirementYearBlock(year=i) for i in range(T_LR, T_D+1) })


    


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

    
    i_ER = T_ER -AGE_0 
    i_LR =  T_LR - AGE_0 
    i_D = T_D - AGE_0 +1
    

    all_a = torch.zeros(B,i_D+1 ).to(device)
    all_h = torch.zeros(B, i_D).to(device)
    all_y = torch.zeros(B, i_LR).to(device)
    all_c = torch.zeros(B, i_D).to(device)
    all_pr_bar = torch.zeros(B, i_LR - i_ER+1)
    
    
    
    #using the block for the first year, predicting a_2 and h_1
    h_t, x_t = self.work_block[f'year_22'](theta[:, 0], edu, a_1) 
    y_t = all_w[:,0] * h_t
    a_t = (1.0-x_t)*(y_t - social_security_tax(y_t)+ a_1)*(1+R)
    c_t = (x_t)*(y_t - social_security_tax(y_t)+ a_1)+1e-8
    
    
    
    all_a[:,0] = a_1
    all_h[:,0] = h_t
    all_y[:,0] = y_t
    all_c[:, 0] = c_t
    all_a[:,1] = a_t
    
    
    # loop over years until early retirement,
    # In each year, previously generated values of 'a_t' and the corresponding theta will be fed to the model.
    for i in range(1,i_ER):
      

      h_t, x_t = self.work_block[f'year_{i+AGE_0}'](theta[:, i], edu, a_t, all_y[:, :i])
      
      
      y_t = all_w[:,i] * h_t
      all_c[:, i] = (x_t)*(y_t - social_security_tax(y_t)+ a_t) +1e-8
      a_t = (1.0 -x_t)*((y_t) - social_security_tax(y_t) + a_t)* (1+R) 
      all_y[:, i] = y_t
      all_h[:, i] = h_t
      all_a[:,i+1] = a_t
    


    
    

    
    #the probability of becoming retired at year 61 (it is zero)
    pr_bar= torch.zeros_like(a_t)
    
    
    #benefit if retire at age 62
    b_bar = torch.zeros_like(a_t)
    
    for i in range(i_ER, i_LR):
      
      # t_t = torch.ones_like(a_t) * (i+AGE_0)
      benefit = retirement_benefit(all_y[:, :i], 0, 35)

      # f forward(self, theta, edu, a_t, all_y, w_t, b_t, pr_bar_t, b_bar_t):
      
      outputs = self.work_retirement_block[f'year_{i+AGE_0}'](theta[:, i], edu, all_a[:, i], all_y[:, :i,],all_w[:, i], benefit, pr_bar, b_bar)

      #if we are retired (pr = 1) the working hour should be 0
      all_h[:, i] = outputs['h_t']
      all_pr_bar[:, i - i_ER+1] = outputs['pr_bar_tp']
      all_y[:, i] = outputs['y_t']
      all_a[:, i+1] = outputs['a_tp']
      all_c[:, i] = outputs['c_t']
      
      pr_bar = outputs['pr_bar_tp']
      b_bar = outputs['b_bar_tp']
      
      
      
      


      
      
      
      
  

    for i in range(i_LR, i_D):
      
    
      # t_t = torch.ones_like(a_t) * (i+AGE_0)
      
      x_t = self.retirement_block[f'year_{i+AGE_0}'](all_a[:, i], benefit)
      c_t = (x_t *(benefit + a_t)) + 1e-8
  
      all_a[:,i+1] =  ((1.0-x_t)*(benefit + a_t)*(1+R))
      all_c[:,i] = c_t

    
    return  all_a, all_c, all_pr_bar, all_h, all_y
    
      
      
      
      
    

    