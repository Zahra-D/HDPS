import torch
import torch.nn as nn
import torch.nn.functional as F 
from Parameters import *
from functions import *



  

class YearBlock(nn.Module):

  def __init__(self, input_dim=1, num_hidden_node = 10, mode = 'working_year'):
    super().__init__()
    
    self.bn_input = nn.BatchNorm1d(input_dim)
    
    #it could be either working_year or working_retirement_year
    self.mode = mode
    
    if self.mode == 'working_retirement_year':
      self.p_bn = nn.BatchNorm1d(1)
      self.task_layer_r = nn.Linear(num_hidden_node+1, num_hidden_node)
      torch.nn.init.xavier_uniform_(self.task_layer_r.weight)
      # torch.nn.init.xavier_uniform_(self.task_layer_r.bias)
      
      self.year_layer_r = nn.Linear(num_hidden_node, 1)
      torch.nn.init.xavier_uniform_(self.year_layer_r.weight)
      # torch.nn.init.xavier_uniform_(self.year_layer_r.bias)
      
      self.activation_retirement_decision = nn.Sigmoid()
    
    #initializing the layers
    #first two layers of NN that are  general layers
    self.general_layer_1 = nn.Linear(input_dim, num_hidden_node)
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
    
    





  def forward(self, theta, edu, a, y = None, p=None):
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
      # a+=1e-8
      x = torch.concat([theta, edu, a], dim = -1)
    
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

    if self.mode == 'working_retirement_year':
      x_p = self.p_bn(p)
      x = torch.concat([x, x_p], dim=-1)
      x_r = self.task_layer_r(x)
      x_r = self.activation_function(x_r)
      x_r = self.year_layer_r(x_r)
      x_r = self.activation_retirement_decision( 1000 * x_r)

      return x_h, x_x, x_r

    return x_h, x_x


      
 
  
  
class RetirementYearBlock(nn.Module):
    
    def __init__(self, num_hidden_unit = 5):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        
        self.layer_1 = nn.Linear(3, num_hidden_unit)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        # torch.nn.init.xavier_uniform_(self.layer_1.bias)
        
        # self.layer_2 = nn.Linear(num_hidden_unit, num_hidden_unit)
        # torch.nn.init.xavier_uniform_(self.layer_2.weight)
        # # torch.nn.init.xavier_uniform_(self.layer_2.bias)
        
        self.layer_2 = nn.Linear(num_hidden_unit, 1)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        # torch.nn.init.xavier_uniform_(self.layer_3.bias)
        
        
        self.activation_function = nn.GELU()
        self.x_activation = nn.Sigmoid()
        
        
        
    def forward(self, t, a, b):
      x = torch.stack([t, a, b], dim = -1)
      x = self.bn(x)
      
      x = self.layer_1(x)
      x = self.activation_function(x)
      
      x = self.layer_2(x)
      x = self.x_activation(x)
      
 
      
      return x.squeeze()
        




    
  



class Model_retirement(nn.Module):
  


  def __init__(self, T_LR= 70, T_ER= 62, T_D = 82,   num_hidden_node = 10):
    """
    Constructor for the Model class.

    Args:
        N (int): Number of Year blocks in the model. Number of Year Blocks that have retiremnet decision
        N_R (int): 
    """



    super().__init__()

    self.blocks_wr = nn.ModuleDict({ f'year_{i}': YearBlock(i - AGE_0+3, num_hidden_node = num_hidden_node, mode= 'working_year' if  i < T_ER  else 'working_retirement_year') for i in range(AGE_0, T_LR) })
    self.blocks_r = nn.ModuleDict({ f'year_{i}': RetirementYearBlock() for i in range(T_ER, T_D+1) })
 
    


  def forward(self, theta, edu, a_1, w_t):
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


    theta = theta.unsqueeze(dim=-1)
    B = theta.shape[0]
    device = theta.device

    
    i_ER = T_ER -AGE_0 
    i_LR =  T_LR - AGE_0 
    i_D = T_D - AGE_0 +1
    
    #each element show the input asset for the corresponding year, (not a_t+1)
    working_a = torch.zeros(B,i_ER ).to(device)
    
    all_h = torch.zeros(B, i_D).to(device)
    all_y = torch.zeros(B, i_LR).to(device)
    
    #age 22 until 61 (40 in total)
    working_c = torch.zeros(B, i_ER).to(device)
    
    
    
    #using the block for the first year, predicting a_2 and h_1
    h_t, x_t = self.blocks_wr[f'year_22'](theta[:, 0], edu.unsqueeze(1), a_1.unsqueeze(1))
    x_t = x_t.squeeze() 
    y_t = w_t[:,0] * h_t
    a_t = (1.0-x_t)*(y_t - social_security_tax(y_t)+ a_1)*(1+R)
    
    
    
    working_a[:,0] = a_1
    all_h[:,0] = h_t
    all_y[:,0] = y_t

    working_c[:, 0] = (x_t)*(y_t - social_security_tax(y_t)+ a_1)+1e-8
    
    
    # loop over years until early retirement,
    # In each year, previously generated values of 'a_t' and the corresponding theta will be fed to the model.
    for i in range(1,i_ER):
      
      
 
      working_a[:,i] = a_t
      
      h_t, x_t = self.blocks_wr[f'year_{i+AGE_0}'](theta[:, i], edu.unsqueeze(1), a_t.unsqueeze(1), all_y[:, :i])
      
      
      y_t = w_t[:,i] * h_t
      x_t = x_t.squeeze()
      working_c[:, i] = (x_t)*(y_t - social_security_tax(y_t)+ a_t) +1e-8
      a_t = (1.0 -x_t)*((y_t) - social_security_tax(y_t) + a_t)* (1+R) 
      all_y[:, i] = y_t
      all_h[:, i] = h_t
    
    
    retirement_working_c = torch.zeros(B,i_LR - i_ER, 2).to(device)
    retirement_working_a = torch.zeros(B,i_LR - i_ER + 1, 2).to(device)
    retirement_working_pr = torch.zeros(B,i_LR - i_ER).to(device)

    
    
    #set asset of the year 62
    retirement_working_a[:, 0, 0] = a_t
    retirement_working_a[:,0, 1] = a_t
    
    #the probability of becoming retired at year 61 (it is zero)
    pr= torch.zeros_like(a_t)
    
    
    #benefit if retire at age 62
    benefit = retirement_benefit(all_y[:, :i], 0, 35)
    
    for i in range(i_ER, i_LR):
      
      t_t = torch.ones_like(a_t) * (i+AGE_0)
      
      h_t, x_t_w, pr_cur = self.blocks_wr[f'year_{i+AGE_0}'](theta[:, i], edu.unsqueeze(1), a_t.unsqueeze(1), all_y[:, :i,], pr.unsqueeze(1))
      x_t_r = self.blocks_r[f'year_{i+AGE_0}'](t_t, a_t, benefit)
      
      
      #if pr=0 it means  until this year we were still working, so the pr for current year will be define by pr_cur
      #if pr=1 it means we are already retired, so the pr should be passed to further years as it is.
      #based on resulted pr we will decide how to calculate utility for corresponding year. 
      pr = (1-pr) * pr_cur.squeeze(1) + pr
      
      
      
      
      
      # we normally calculate the output of the both states, retirement and working,
      # but for the next year asset input (since we need one single input), we calculate a weighted sum of the output working and retirement asset based on pr. 
      
      
      #working outputs
      y_t = w_t[:,i] * h_t  
      x_t_w = x_t_w.squeeze()
      c_t_w = x_t_w * (y_t - social_security_tax(y_t) + a_t) + 1e-8
      a_t_next_w = (1.0 -x_t_w)*((y_t) - social_security_tax(y_t) + a_t)* (1+R) 
      
      #retirement outputs
      a_t_next_r = ((1.0-x_t_r)*(benefit + a_t)*(1+R))
      c_t_r = (x_t_r *(benefit + a_t))+ 1e-8
      
      

      #index 0 for working and 1 for retirement
      retirement_working_a[:,i - i_ER+1, 0] = a_t_next_w
      retirement_working_a[:,i - i_ER+1, 1] = a_t_next_r
      retirement_working_c[:, i - i_ER, 0] = c_t_w
      retirement_working_c[:, i - i_ER, 1] = c_t_r
      
      #if we are retired (pr = 1) the working hour should be 0
      all_h[:, i] = (1-pr) * h_t
      retirement_working_pr[:, i - i_ER] = pr
      
      
      
      all_y[:, i] = (1-pr)  *  y_t
      
      #next year benefit and asset
      a_t = (1-pr) * a_t_next_w + pr * a_t_next_r  
      benefit = (1-pr) * benefit + (pr) * retirement_benefit(all_y[:, :i], i-i_ER, 35)
      
      
      
      
      
      
      
      
      
      
    retirement_a = torch.zeros(B,i_D - i_LR+1).to(device)
    retirement_c = torch.zeros(B, i_D - i_LR).to(device)
    retirement_a[:, 0] = a_t

    for i in range(i_LR, i_D):
      
    
      t_t = torch.ones_like(a_t) * (i+AGE_0)
      
      x_t = self.blocks_r[f'year_{i+AGE_0}'](t_t, a_t, benefit)
      c_t = (x_t *(benefit + a_t)) + 1e-8
      
      a_t =  ((1.0-x_t)*(benefit + a_t)*(1+R))
      retirement_a[:,i- i_LR+1] = a_t
      retirement_c[:,i- i_LR] = c_t

    
    return retirement_working_pr, working_c, retirement_working_c, retirement_c, working_a, retirement_working_a, retirement_a, all_h, all_y
    
      
      
      
      
    



