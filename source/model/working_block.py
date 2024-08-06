from source.utils.imports import * 
from .decorators import dimension_corrector
from .task_block import TaskBlock
from source.economic import Economic






class WorkYearBlock(nn.Module):

  def __init__(self, num_input=1,
               num_hidden_node = 10,
               mode = 'working_year',
               alpha_h = 1,
               alpha_pr= None,
               hard_gumbel = True,
               layers_dict=None):
    super().__init__()
    
    gen_2 = nn.Linear(num_hidden_node,num_hidden_node)
    torch.nn.init.xavier_uniform_(gen_2.weight)

    general_layer_2 = layers_dict['general2'] if 'general2' in layers_dict else gen_2
    task_layer_h = layers_dict['task_h'] if 'task_h' in layers_dict else None
    task_layer_a_w = layers_dict['task_aw'] if 'task_aw' in layers_dict else None
    task_layer_a_r = layers_dict['task_ar'] if 'task_ar' in layers_dict else None
    task_layer_pr = layers_dict['task_pr'] if 'task_pr' in layers_dict else None
    
    
    # if mode == "early_retirement_year":
    #   num_input += 1 # there should be benefit as well
    self.bn_input = nn.BatchNorm1d(num_input)
    
    #it could be either working_years or early_retirement_years
    self.mode = mode
    self.hard_gumbel = hard_gumbel
    self.activation_function = nn.GELU()
    
    self.alpha_h = alpha_h
    
    #initializing the layers
    #first two layers of NN that are  general layers
    self.general_layer_1 = nn.Linear(num_input, num_hidden_node)
    torch.nn.init.xavier_uniform_(self.general_layer_1.weight)


    self.general_layer_2 = general_layer_2





 
    self.task_a_w = TaskBlock(num_hidden_node=num_hidden_node, num_output=1, activation_funcion=self.activation_function, task_layer=task_layer_a_w)

    # # h is categorical with 4 categories and for the final result we will get mean of all the four output
    self.task_h = TaskBlock(num_hidden_node=num_hidden_node, num_output=4, activation_funcion=self.activation_function,task_layer=task_layer_h )




    # activation function, we can use a single ReLU wherever it is needed.

    self.a_activation = nn.Sigmoid()
    self.gumbel = F.gumbel_softmax
    # self.h = torch.tensor([0.0,1300.0,2080.0,2860.0])
    
    
    if mode == 'early_retirement_year':
          
          self.task_a_r = TaskBlock(num_hidden_node=num_hidden_node, num_output=1, activation_funcion=self.activation_function, task_layer=task_layer_a_r)
          
          #r fr fr  free (early_retirement 10)
          self.alpha_pr = alpha_pr
          
          self.task_pr = TaskBlock(num_hidden_node=num_hidden_node, num_output=2,  activation_funcion=self.activation_function, task_layer=task_layer_pr)
      
    
    




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



    
    
    if isinstance(y, torch.Tensor):
      if isinstance(b, torch.Tensor):
        x = torch.concat([theta, edu, a, y, b], dim = -1)
      else:
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



    x_x_w = self.task_a_w(x)
    x_x_w = self.a_activation(x_x_w)

    
    




    x_h = self.task_h(x)
    # x_h = (self.a_activation(x_h) * self.h).squeeze(-1)
    # x_h = F.softmax(x_h, dim=-1)
    
    x_h = self.gumbel( x_h, hard=self.hard_gumbel, tau=self.alpha_h)
    

     
    # one_hot[torch.arange(B).to(device), torch.argmax(x_h, dim = 1)] = self.h[torch.argmax(x_h, dim = 1)]
    x_h = torch.einsum('bh,h->b',x_h, Economic.H.to(device))
    # x_h = x_h * self.h



    if self.mode == 'early_retirement_year':


    
    
      x_x_r = self.task_a_r(x)
      x_x_r = self.a_activation(x_x_r)
      # print(F.softmax(self.task_pr(x), dim= -1).shape)
      # logit = torch.log(F.softmax(self.task_pr(x), dim= -1))
      pr = self.gumbel( self.task_pr(x), hard=self.hard_gumbel, tau=self.alpha_pr)
      # print(self.hard_gumbel)
      # pr = self.a_activation(self.alpha_pr * self.task_pr(x))
      return x_h, x_x_w.squeeze(), pr, x_x_r.squeeze()
      



    return x_h, x_x_w.squeeze()
  