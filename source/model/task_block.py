
from source.utils.imports import * 


class TaskBlock(nn.Module):

  def __init__(self, num_hidden_node=10, num_output = 1, activation_funcion = nn.GELU(), task_layer = None):
    super().__init__()
        # third layer which is task specific layer, one for h and one for a
    # third layer which is task specific layer, one for h and one for a
    
    if not task_layer:
      self.task_layer = nn.Linear(num_hidden_node,num_hidden_node, bias=True)
      torch.nn.init.xavier_uniform_(self.task_layer.weight)
    else:
      self.task_layer = task_layer 
    
    
    #forth layer that dedicated to year, again one for h and one for a
    self.year_layer= nn.Linear(num_hidden_node,num_output)
    torch.nn.init.xavier_uniform_(self.year_layer.weight)
    
    self.activation_function = activation_funcion
    

    

  def forward(self, x):
        
    x = self.task_layer(x)
    x = self.activation_function(x)
    x = self.year_layer(x)
    return x

    