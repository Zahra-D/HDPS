
import torch

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
