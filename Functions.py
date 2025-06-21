### Defining functions used in the model


from imports import *
from Parameters import *


## Wage Function --------------------


# Deterministic part
wage_det = lambda edu, t: BETA_w_0 + BETA_w_1 * (t+AGE_0) +  BETA_w_2 * (t+AGE_0)**2 + BETA_w_3 * edu + BETA_w_4 * edu * (t+AGE_0) + BETA_w_5 * edu * (t+AGE_0)**2
# Stochastic part 
e = lambda n: torch.normal(0, SIGMA_e, n) # Shcoks to persistant part
theta = lambda theta_pre,e: theta_pre + e   # Update the persistant part (Markov d1)
# Wage (stochstic part+determisnitic part), accouting for min wage
wage = lambda mu_t, theta_t : torch.maximum(torch.e**(mu_t + theta_t) , torch.tensor(w_min))


## Budget Constaint --------------------


consumption = lambda a_t, y_t : a_t[:, :-1] + y_t -  a_t[:, 1:]/(1+R)


## Public Policies --------------------


income_tax  = lambda y_t: y_t - KAPPA_Inc_Tax * (y_t+1e-8)**(1-TAU_Inc_Tax)

social_security_tax = lambda y_t: TAU_SS_Tax * torch.minimum(y_t, Inc_base_SS)

# Score = lambda y_t, T_S: torch.topk(y_t, T_S).values.mean(dim=-1)

# pension_benefit = lambda S: ((BEND_POINTS_b_SS[0] * S * (S <= BEND_POINTS_a_SS[0])) +
#                              (BEND_POINTS_b_SS[0] * BEND_POINTS_a_SS[0] + BEND_POINTS_b_SS[1] * (S - BEND_POINTS_a_SS[0])) * ((BEND_POINTS_a_SS[0] < S) & (S <= BEND_POINTS_a_SS[1])) +
#                              (BEND_POINTS_b_SS[0] * BEND_POINTS_a_SS[0] + BEND_POINTS_b_SS[1] * (BEND_POINTS_a_SS[1] - BEND_POINTS_a_SS[0]) + BEND_POINTS_b_SS[2] * (S - BEND_POINTS_a_SS[1])) * ((BEND_POINTS_a_SS[1] < S) & (S <= BEND_POINTS_a_SS[2])) +
#                              (BEND_POINTS_b_SS[0] * BEND_POINTS_a_SS[0] + BEND_POINTS_b_SS[1] * (BEND_POINTS_a_SS[1] - BEND_POINTS_a_SS[0]) + BEND_POINTS_b_SS[2] * (Inc_base_SS - BEND_POINTS_a_SS[1])) * (S > BEND_POINTS_a_SS[2]))

# def retirement_benefit(all_y, t_R, TS=35):
#     delta_t = DELTA_t_SS.to(all_y.device)
#     delta = delta_t[t_R]
#     S = Score(all_y, TS)
#     b = pension_benefit(S) * delta
#     return b

def compute_pension_benefit(S):
    b = 0
    a = BEND_POINTS_a_SS
    c = BEND_POINTS_b_SS
    # First segment
    b += c[0] * torch.minimum(S, a[0])
    # Second segment
    b += c[1] * torch.clamp(S - a[0], min=0, max=a[1] - a[0])
    # Third segment
    b += c[2] * torch.clamp(S - a[1], min=0, max=a[2] - a[1])
    # Any income beyond max SS income does not earn more benefit
    return b

def retirement_benefit(all_y, t_R, SS_Type, SS_Param):
    """
    all_y: [B, T] tensor of income paths
    SS_Type: 'Top_Years' or 'Life_Time'
    SS_Param: if Top_Years, number of best years (e.g. 35)
    """
    delta_t = DELTA_t_SS.to(all_y.device)
    b = None

    if SS_Type == "Top_Years":
        S = torch.topk(all_y, SS_Param, dim=1).values.mean(dim=1)  # Mean of top years
    elif SS_Type == "Life_Time":
        S = all_y.mean(dim=1)  # Mean of all years (even zeros)

    pension = compute_pension_benefit(S)

    # Assume earliest retirement age for now (can be extended easily)
    b = pension * delta_t[t_R]  # Assume retiring at 62 = t=6

    return b


## Utility --------------------

def utility_fixed_retirement(c_t, h_t, args):
    """
    Compute discounted utility over the life cycle (fixed retirement age).

    Args:
        c_t: [B, T_D - AGE_0] consumption
        h_t: [B, T_D - AGE_0] hours worked
        args: contains phi, psi, etc.

    Returns:
        total_utility: [B] lifetime utility
    """
    device = c_t.device
    T = c_t.shape[1]
    BETA_t = torch.pow(BETA, torch.arange(T, device=device))  # [T]

    # Utility of consumption
    cons_utility = (c_t ** (1 - GAMMA)) / (1 - GAMMA)  # [B, T]
    #cons_utility = torch.log(c_t)  # [B, T]
    # Disutility of working hours
    disutility_hours = ((h_t/h_max) ** (1 + 1 / ETA)) / (1 + 1 / ETA)  # [B, T]
    #disutility_hours = torch.log(h_t/h_max)
    # Disutility of labor supply
    disutility_work = (h_t > 0).float()  # indicator if working

    period_util = BETA_t * (cons_utility - args.psi * disutility_hours - args.phi * disutility_work)  # [B, T]
    return period_util.sum(dim=1)  # total utility [B]


## Loss Function --------------------


def loss_function_fixed_retirement(c_t, h_t, epoch, s_writer, args):
    """
    Computes the loss for the fixed-retirement model with no regularization.

    Args:
        c_t: [B, T] simulated consumption
        h_t: [B, T] simulated hours
        epoch: current epoch (for logging)
        s_writer: tensorboard summary writer
        args: contains utility parameters (phi, psi)

    Returns:
        loss: scalar tensor
    """
    util = utility_fixed_retirement(c_t, h_t, args)  # [B] lifetime utility per individual

    # Loss is negative average utility
    loss = -1.0 * util.mean()

    # Logging
    # s_writer.add_scalar('Loss/util_term', util.mean().detach().cpu(), epoch)
    # s_writer.add_scalar('Loss/total', loss.item(), epoch)

    return loss


## Generating the simulated exogenous state vars dataset  --------------------

  
def generating_dataset(number_samples, duration, theta_0, p_edu):
  
    ep_t = e((number_samples, duration))
    theta_t = torch.cumsum(ep_t, dim=-1) + theta_0
    prob = torch.tensor([p_edu] * number_samples)
    edu = torch.bernoulli(prob)
    u_t = wage_det(edu.unsqueeze(1), torch.arange(1,duration+1))
    w_t = wage(u_t, theta_t)

    return TensorDataset(theta_t, w_t, edu)
   
  
## Run Analyis --------------------


def save_checkpoint(model, optimizer, base_dir, epoch):
    
    pathlib.Path(f'{base_dir}/epoch{epoch}').mkdir(parents=True, exist_ok=True) # Folder of a run and epoch
           
    torch.save(optimizer.state_dict(), f'{base_dir}/epoch{epoch}/optimizer_state.pth') # Save optimzer's state
    
    rng_checkpoint = { # Random seed
        'torch_rng_state':
            torch.get_rng_state(), # PyTorch’s RNG (CPU).
        'cuda_rng_state':
            torch.cuda.get_rng_state_all(), # PyTorch’s CUDA RNG (for GPU).
        'numpy_rng_state':
            np.random.get_state(), # NumPy’s random state.
        'python_rng_state':
            random.getstate() # Python’s built-in random module state.
    }
    with open(f'{base_dir}/epoch{epoch}/rng_checkpoint.pkl', 'wb') as f:
        pickle.dump(rng_checkpoint, f) # Save Random Seeds
    
    torch.save(model, f"{base_dir}/epoch{epoch}/model.pt") # Save Model architercture and parameteres


