##### Defninng DNN model --------------------


from imports import *
from Parameters import *
from functions import *


## Helper Functions  --------------------


# Avoid Dimension mismatch: Ensures that all input tensors have at least two dimensions (1D -> 2D if needed)

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


## Task-Specific DNN Block --------------------


class TaskBlock(nn.Module):

  def __init__(self, num_hidden_node=10, num_output = 1, activation_funcion = nn.GELU(), task_layer = None):
    super().__init__()

    # task_layer: Task specific layer category
    
    if not task_layer:
      # If task_layer is not provided, a linear layer is defined.
      self.task_layer = nn.Linear(num_hidden_node,num_hidden_node, bias=True)
      torch.nn.init.xavier_uniform_(self.task_layer.weight)
    else:
      # If task_layer is provided, it replaces the default task-specific layer.
      self.task_layer = task_layer 
       
    # year_layer: Task-Age specific layer category is added to task_layer
    
    self.year_layer= nn.Linear(num_hidden_node,num_output)
    torch.nn.init.xavier_uniform_(self.year_layer.weight)
    self.activation_function = activation_funcion

  # Forward Pass: DNN Model -> Task_Layer -> Activation FUnction -> Year_Layer (w. GELU AF)
  def forward(self, x):
        
    x = self.task_layer(x)
    x = self.activation_function(x)
    x = self.year_layer(x)
    return x

  
## Working Ages Block --------------------

  
  # Also define Early retienemt ages as mode of this block, later define its foraward using this.
    
class WorkYearBlock(nn.Module):

  def __init__(self, num_input=1, num_hidden_node = 10, mode = 'working_year', alpha_pr= 1, layers_dict=None):
    super().__init__()

    # Working Years: All Layers Ater First

    # Types of Layers

    # Batch normalization layer
    self.bn_input = nn.BatchNorm1d(num_input)
    # General layer 2
    gen_2 = nn.Linear(num_hidden_node,num_hidden_node)     
    torch.nn.init.xavier_uniform_(gen_2.weight)
    general_layer_2 = layers_dict['general2'] if 'general2' in layers_dict else gen_2
    # Work Hour task, conditional on wokring this period
    task_layer_h = layers_dict['task_h'] if 'task_h' in layers_dict else None     
    # Asset next period, conditional on working this period
    task_layer_a_w = layers_dict['task_aw'] if 'task_aw' in layers_dict else None 
    # Asset next period, conditional on retiring this period
    task_layer_a_r = layers_dict['task_ar'] if 'task_ar' in layers_dict else None
    # Probaility of becoming retired this period
    task_layer_pr = layers_dict['task_pr'] if 'task_pr' in layers_dict else None
    
    # Activation Functions

    self.activation_function = nn.GELU()
    self.a_activation = nn.Sigmoid() # For Asseet (0 < a/Total resource  < 1)
    self.h_activation = nn.Sigmoid() # For Continiuos hours of work (0 < h  < 1)
    self.gumbel = F.gumbel_softmax # For retiermnet (Discrte choice)
    #self.h = h_grid # For h grid

    # Mode: eaither working_years or early_retirement_years (based on input) 
    
    self.mode = mode
    
    # Building the Layers Architecture

    # General Layers: general_layer_1,2
    self.general_layer_1 = nn.Linear(num_input, num_hidden_node)
    torch.nn.init.xavier_uniform_(self.general_layer_1.weight)
    self.general_layer_2 = general_layer_2
    # Task a_w: Task Layer Asset, conditional on working: task_a_w
    self.task_a_w = TaskBlock(num_hidden_node=num_hidden_node, num_output=1, activation_funcion=self.activation_function, task_layer=task_layer_a_w)
    # Task h: Hours of work (categorical with 4 categories and for the final result we will get mean of all the four output)
    # self.task_h = TaskBlock(num_hidden_node=num_hidden_node, num_output=4, activation_funcion=self.activation_function,task_layer=task_layer_h)
    self.task_h = TaskBlock(num_hidden_node=num_hidden_node, num_output=1, activation_funcion=self.activation_function,task_layer=task_layer_h)
    # Extra Task Layer for Early retierment Years
    if mode == 'early_retirement_year':
          # Task a_w: Task Layer Asset, conditional on becoming retierd
          self.task_a_r = TaskBlock(num_hidden_node=num_hidden_node, num_output=1, activation_funcion=self.activation_function, task_layer=task_layer_a_r)
          # Task pr: probailibity of beconimg retierd
          #self.alpha_pr = alpha_pr # Paramter of f(pr)
          self.task_pr = TaskBlock(num_hidden_node=num_hidden_node, num_output=2,  activation_funcion=self.activation_function, task_layer=task_layer_pr)

  # Forward Pass: x -> General Layers of 1 & 2 -> x_x_w,x_h(,x_x_r,pr)

  @dimension_corrector
  def forward(self, theta, edu, a, y = None, b = None):
    
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

    # Adjusting number of state variables
    if isinstance(y, torch.Tensor):
      if isinstance(b, torch.Tensor):
        x = torch.concat([theta, edu, a, y, b], dim = -1)
      else:
        x = torch.concat([theta, edu, a, y], dim = -1)
    else:
      x = torch.concat([theta, edu, a], dim = -1)

    # Device
    device = x.device

    # General Layers
    x = self.bn_input(x)
    x = self.general_layer_1(x)
    x = self.activation_function(x)
    x = self.general_layer_2(x)
    x = self.activation_function(x)
    # Task Layer: 1-a_w'/resource if work cond. on being in worker state
    x_w_w = self.task_a_w(x)
    x_w_w = self.a_activation(x_w_w)
    # Task Layer: h
    x_h = self.task_h(x)
      # Method. Continous h
    x_h = self.h_activation(x_h)
      # Method. Discrte h
    #x_h = F.softmax(x_h, dim=-1)
    #self.h = self.h.to(device)
      # Method. Use weighted mean as approximation
    #x_h = torch.einsum('bh,h->b',x_h, self.h) 
      # Method. Use Max pf probabilities
    # one_hot[torch.arange(B).to(device), torch.argmax(x_h, dim = 1)] = self.h[torch.argmax(x_h, dim = 1)] 
    # x_h = x_h * self.h
    if self.mode == 'early_retirement_year':
      # Task Layer: 1-a_r'(/resource) if retier cond. on being in worker state
      x_r_w = self.task_a_r(x)
      x_r_w = self.a_activation(x_r_w)
      # Task Layer: x_pr probability of retieremnt conditional on being in worker state
        # Method. Soft & Hard Gambul
      x_pr = self.gumbel(self.task_pr(x), hard=True)
        # Method. Temperute Parameter?
      # pr = self.a_activation(self.alpha_pr * self.task_pr(x)) 
      return x_h.squeeze(), x_w_w.squeeze(), x_pr, x_r_w.squeeze()
      
    return x_h.squeeze(), x_w_w.squeeze()
  

## Retierment Ages Block --------------------


class RetirementYearBlock(nn.Module):
    
    def __init__(self, num_hidden_unit = 5):
        super().__init__()

        # Layers

          # Method. Take year as input, same DNN for all years
        # Batch normalization layer
        self.bn = nn.BatchNorm1d(3)
        # General Layers
        self.layer_1 = nn.Linear(3, num_hidden_unit)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        self.layer_2 = nn.Linear(num_hidden_unit, 1)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
          # Method. Different DNN for differenet years
        # self.year = year

        # Activation Functions
      
        self.activation_function = nn.GELU()
        self.x_activation = nn.Sigmoid()

    # Forward Pass: X (asset,pension benefit,year) -> 2 General Layers -> 1-asset(/resource) next year
  
    @dimension_corrector
    def forward(self, a, b, t):
      
      x = torch.concat([a, b, t], dim = -1)
      x = self.bn(x)
      x = self.layer_1(x)
      x = self.activation_function(x)
      x = self.layer_2(x)
      x = self.x_activation(x)
      
      return x.squeeze()


## Early Retierment Ages Block --------------------

  
class EarlyRetiermentBlock(nn.Module):
  
      def __init__(self, year=61,retirement_block=None, num_hidden_node_w = 10, num_hidden_node_r = 5, alpha_pr = 1, layers_dict=None):
        super().__init__()
        
        assert (year >= T_ER) and (year <= T_LR) # The years in which the worker has to take retirement decision
        
        self.working_block = WorkYearBlock(num_input=year - AGE_0 + 3, num_hidden_node = num_hidden_node_w,  mode= 'early_retirement_year', alpha_pr=alpha_pr, layers_dict=layers_dict)
        self.retirement_block = retirement_block        
        self.year= year

      # Forward Pass: Inputs at each year: theta (productivity), edu (education), a_w_t (asset conditioanl on being worker), a_r_t (asset conditioanl on being retierd),
      # all_y_t (vector of income history), w_t (wage), pr_bar_t (probility of being retied before this year), b_bar_t (expected pension benefit if retied before this year)
        
      def forward(self, theta, edu, a_w_t, a_r_t, all_y_t, w_t, pr_bar_t, b_bar_t):

        # Policy given policy functions
        
        h_ww_t, x_ww_t, pr_t, x_rw_t = self.working_block(theta, edu, a_w_t, all_y_t) # Cond. on being a worker: hours of work, 1-asset'/rources if work, probability of retierment, 1-asset'/resource if retierd
        t = torch.ones_like(a_r_t).to(a_r_t.device) * self.year # vector of age
        x_rr_t = self.retirement_block(a_r_t,b_bar_t,t) # conditional on being a retieree: 1-asset'/resouce 

        # Budget Constaint 
        
        # Decide to work conditional on start in worker state
        #w_t = wage(mu(edu,year-Age_0),theta)
        # labor income
          # Method. Continous h
        y_ww_t = w_t * (h_ww_t*h_max)
          # Method. Discrte h
        #y_ww_t = w_t * h_ww_t
        re_ww_t = y_ww_t - income_tax(y_ww_t) - social_security_tax(y_ww_t) + a_w_t
        c_ww_t = x_ww_t*re_ww_t + 1e-8
        a_next_ww_t = (1.0 - x_ww_t)*(1+R)*re_ww_t
        # Decice to retire conditional on start in worker state
        b_t = retirement_benefit(all_y_t, self.year - T_ER, Len_S_Max_SS)
        re_rw_t = b_t + a_w_t
        c_rw_t = x_rw_t*re_rw_t + 1e-8
        a_next_rw_t = (1.0 - x_rw_t)*(1+R)*re_rw_t
        # Start in retire state
        re_rr_t = b_bar_t + a_r_t
        c_rr_t = x_rr_t*re_rr_t + 1e-8
        a_next_rr_t = (1.0-x_rr_t)*(1+R)*re_rr_t
        
        # Next Year state variables

        # Asset next year, cond. being a worker then
        a_next_w_t = a_next_ww_t
        # Asset next year, cond. being a retire then
        a_next_r_t = (1-pr_bar_t)*a_next_rw_t  +  pr_bar_t*a_next_rr_t
        # Update starring the next year as a retire
        pr_bar_next_t = pr_bar_t + (1-pr_bar_t)*pr_t[:,0]
        # Update the expected Pension Benefit the next year cond. starting as a retire
        b_bar_next_t = pr_bar_t*b_bar_t + (1-pr_bar_t)*b_t

        # Expected vars for analysis
        
        a_next_t = (1-pr_bar_next_t)*a_next_w_t + pr_bar_next_t*a_next_r_t
        c_w_t = (1.0 - pr_t[:,0])*c_ww_t + pr_t[:,0]*c_rw_t
        c_t = (1-pr_bar_t)*c_w_t + pr_bar_t*c_rr_t
        h_w_t = (1.0 - pr_t[:,0])*h_ww_t
        h_t = (1-pr_bar_t)*h_w_t    
        y_w_t = (1.0 - pr_t[:,0])*y_ww_t
        y_t = (1-pr_bar_t)*y_w_t      
        
        return {'a_next_ww_t':a_next_ww_t,'a_next_rw_t': a_next_rw_t,'a_next_rr_t': a_next_rr_t,'a_next_t': a_next_t,
                'c_ww_t': c_ww_t,'c_rw_t': c_rw_t,'c_rr_t': c_rr_t,'c_t': c_t,          
                'y_ww_t': y_ww_t,'y_w_t': y_w_t,'y_t': y_t,
                'h_ww_t': h_ww_t,'h_w_t': h_w_t,'h_t': h_t,
                'pr_t': pr_t[:,0],'pr_bar_next_t': pr_bar_next_t,'b_bar_next_t': b_bar_next_t}
         
        
## Full Model (All Blocks) --------------------
        

class Model(nn.Module):

  def __init__(self, T_LR= 70, T_ER= 62, T_D = 82,   num_hidden_node_w = 10, num_hidden_node_r = 5, alpha_pr = 1):
    
    """
    Constructor for the Model class.

    Args:
        N (int): Number of Year blocks in the model. Number of Year Blocks that have retiremnet decision
        N_R (int): 
    """

    super().__init__()

    # Defining DNN for all the years
    
    layers_dict = self.initializing_layer_dict(num_hidden_node_w)
    
    retirement_block = RetirementYearBlock()
    self.work_blocks = nn.ModuleDict({ f'year_{i}': WorkYearBlock(i - AGE_0+3, num_hidden_node = num_hidden_node_w, layers_dict=layers_dict) for i in range(AGE_0, T_ER) })
    self.work_retirement_blocks = nn.ModuleDict({ f'year_{i}': EarlyRetiermentBlock(year=i, retirement_block = retirement_block, num_hidden_node_r=num_hidden_node_r, num_hidden_node_w= num_hidden_node_w, alpha_pr=alpha_pr , layers_dict = layers_dict) for i in range(T_ER, T_LR+1) })
    self.retirement_blocks = retirement_block 
    
  # Dictionary of type of layers used in all blocks
  
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

  # Forward Pass: simulation from the beginng to death
  # Inputs: wage: theta (theta) and wage (all_w) vector, permanent types: edu (edu), initial states: asset (a_1)
  
  def forward(self, theta, edu, A_0, all_w):
    
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

    # Creating Output vars matrix
    
    B = theta.shape[0] # Batch size
    device = theta.device # GPU/CPU
    # Size of each stage: Wokring, retirement decision, retire
    i_ER = T_ER -AGE_0 
    i_LR =  T_LR - AGE_0 
    i_D = T_D - AGE_0 +1
    # Vector of outputs
    all_a = torch.zeros(B,i_D+1 ).to(device)
    all_h = torch.zeros(B, i_D).to(device)
    all_y = torch.zeros(B, i_LR).to(device)
    all_c = torch.zeros(B, i_D).to(device)
    # theta = theta.unsqueeze(dim=-1)
    
    # Year 1

    h_t, x_t = self.work_blocks[f'year_22'](theta[:, 0], edu, A_0) 
    # labor income
        # Method. Continous h
    y_t = all_w[:,0] * (h_t*h_max)
          # Method. Discrte h
    #y_t = all_w[:,0] * h_t
    re_t = y_t - income_tax(y_t) - social_security_tax(y_t) + A_0 
    c_t = x_t*re_t + 1e-8
    a_next_t = (1.0 - x_t)*(1+R)*re_t
    
    all_h[:,0] = h_t
    all_y[:,0] = y_t
    all_c[:,0] = c_t
    all_a[:,0] = A_0
    all_a[:,1] = a_next_t
    all_c_ER = torch.zeros(B, i_LR - i_ER+1, 3).to(device)
    
    # Loop over years until early retirement
    
    for i in range(1,i_ER):
    
      h_t, x_t = self.work_blocks[f'year_{i+AGE_0}'](theta[:, i], edu, all_a[:,i], all_y[:, :i])
          # Method. Continous h
      y_t = all_w[:,i] * (h_t*h_max)
          # Method. Discrte h
      #y_t = all_w[:,i] * h_t
      re_t = y_t - income_tax(y_t) - social_security_tax(y_t) + a_t
      c_t = x_t*re_t + 1e-8
      a_next_t = (1.0 - x_t)*(1+R)*re_t
      
      all_y[:, i] = y_t
      all_h[:, i] = h_t
      all_c[:, i] = c_t
      all_a[:,i+1] = a_next_t 

    # State vars for T_ER
    
    pr_bar_t = torch.zeros_like(a_next_t) # zero pr of starting reiterd at T_ER
    b_bar_t = torch.zeros_like(a_next_t) # zero pension benefit if starting reiterd at T_ER
    a_w_t = a_r_t = a_next_t # Starting asset given work/retire state
    
    all_c_ER = torch.zeros(B, i_LR - i_ER+1, 3).to(device)
    all_pr = torch.zeros(B, i_LR - i_ER+1).to(device)
    all_pr_bar = torch.zeros(B, i_LR - i_ER+1).to(device)

    for i in range(i_ER, i_LR):

      outputs = self.work_retirement_blocks[f'year_{i+AGE_0}'](theta[:,i],edu,a_w_t,a_r_t,all_y[:,:i,],all_w[:,i],pr_bar_t,b_bar_t)

      all_h[:, i] = outputs['h_t']
      all_pr_bar[:, i - i_ER+1] = outputs['pr_bar_next_t']
      all_a[:, i+1] = outputs['a_next_t']
      all_c[:, i] = outputs['c_t'] 
      all_c_ER[:, i-i_ER, 0] = outputs['c_ww_t'] # index 0 for ww
      all_c_ER[:, i-i_ER, 1] = outputs['c_rw_t'] # index 1 for rw
      all_c_ER[:, i-i_ER, 2] = outputs['c_rr_t'] # index 2 for rr
      all_pr[:,i-i_ER] =  outputs['pr_t']
      all_pr_bar[:,i-i_ER+1] =  outputs['pr_bar_next_t']
      
      all_y[:,i] = outputs['y_ww_t']
      pr_bar_t = outputs['pr_bar_next_t']
      b_bar_t = outputs['b_bar_next_t']
      a_w_t = outputs['a_next_w_t']
      a_r_t = outputs['a_next_r_t']
      
    # The Latest retiement year
    
    outputs = self.work_retirement_blocks[f'year_{i_LR+AGE_0}'](theta[:, i_LR-1],edu,a_w_t,a_r_t,all_y[:,:i_LR,],all_w[:, i_LR-1],pr_bar_t, b_bar_t)

    all_h[:, i_LR] = 0.00
    all_y[:, i_LR] = 0.00
    all_a[:, i_LR+1] = outputs['a_next_r_t']
    all_c[:, i_LR] = (1-pr_bar_t)*outputs['c_rw_t'] + pr_bar_t*outputs['c_rr_t']
    all_c_ER[:, i_LR-i_ER+1, 0] = 1e-8
    all_c_ER[:, i_LR-i_ER+1, 1] = outputs['c_rw_t']
    all_c_ER[:, i_LR-i_ER+1, 2] = outputs['c_rr_t']
    b_bar = outputs['b_bar_next_t']
    all_pr[:,i_LR-i_ER+1] =  1

    # The retiremnt years

    for i in range(i_LR+1, i_D):
      
      t = torch.ones_like(a_r_t).to(a_r_t.device) * (i+AGE_0)
      x_t = self.retirement_blocks(all_a[:,i],b_bar,t)
      re_t = b_bar + a_r_t
      c_t = x_t*re_t + 1e-8
      a_next_t = (1.0 - x_t)*(1+R)*re_t
      all_a[:,i+1] = a_next_t
      all_c[:,i] = c_t
      all_h[:, i] = 0.00
  
    return  all_a, all_c, all_c_ER, all_pr_bar, all_pr, all_h, all_y
