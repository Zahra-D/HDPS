### Defining functions used in the model


from imports import *
from Parameters import *


## Wage Function --------------------


# Deterministic part
mu = lambda edu, t: BETA_w_0 + BETA_w_1 * (t+AGE_0) +  BETA_w_2 * (t+AGE_0)**2 + BETA_w_3 * edu + BETA_w_4 * edu * (t+AGE_0) + BETA_w_5 * edu * (t+AGE_0)**2
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
Score = lambda y_t, T_S: torch.topk(y_t, T_S).values.mean(dim=-1)

pension_benefit = lambda S: ((BEND_POINTS_b_SS[0] * S * (S <= BEND_POINTS_a_SS[0])) +
                             (BEND_POINTS_b_SS[0] * BEND_POINTS_a_SS[0] + BEND_POINTS_b_SS[1] * (S - BEND_POINTS_a_SS[0])) * ((BEND_POINTS_a_SS[0] < S) & (S <= BEND_POINTS_a_SS[1])) +
                             (BEND_POINTS_b_SS[0] * BEND_POINTS_a_SS[0] + BEND_POINTS_b_SS[1] * (BEND_POINTS_a_SS[1] - BEND_POINTS_a_SS[0]) + BEND_POINTS_b_SS[2] * (S - BEND_POINTS_a_SS[1])) * ((BEND_POINTS_a_SS[1] < S) & (S <= BEND_POINTS_a_SS[2])) +
                             (BEND_POINTS_b_SS[0] * BEND_POINTS_a_SS[0] + BEND_POINTS_b_SS[1] * (BEND_POINTS_a_SS[1] - BEND_POINTS_a_SS[0]) + BEND_POINTS_b_SS[2] * (Inc_base_SS - BEND_POINTS_a_SS[1])) * (S > BEND_POINTS_a_SS[2]))

def retirement_benefit(all_y, t_R, TS=35):
    delta_t = DELTA_t_SS.to(all_y.device)
    delta = delta_t[t_R]
    S = Score(all_y, TS)
    b = pension_benefit(S) * delta
    return b
