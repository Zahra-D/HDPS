from source.utils.imports import *
from .decorators import dimension_corrector


class RetirementYearBlock(nn.Module):
    
    def __init__(self, num_hidden_unit = 5, year=75):
        super().__init__()
        

        
        self.year = year
        
        self.bn = nn.BatchNorm1d(2)
        
        self.layer_1 = nn.Linear(2, num_hidden_unit)
        torch.nn.init.xavier_uniform_(self.layer_1.weight)

        
        self.layer_2 = nn.Linear(num_hidden_unit, 1)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)

        
        
        self.activation_function = nn.GELU()
        self.x_activation = nn.Sigmoid()
        
        
    @dimension_corrector
    def forward(self, a, b):
      x = torch.concat([a, b], dim = -1)
      x = self.bn(x)
      
      x = self.layer_1(x)
      x = self.activation_function(x)
      
      x = self.layer_2(x)
      x = self.x_activation(x)
      

      
 
      
      return x.squeeze()
        
   