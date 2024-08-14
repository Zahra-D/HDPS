from imports import *


from Parameters import *


#functions

#u_t(EDU, t): calculate u_t for the inputs EDU and t, if t=[1,2,3,...] => u_t=[u_1, u_2, u_3, ...]
mu = lambda edu, t: BETA_w_0 + BETA_w_1 * (t+AGE_0) +  BETA_w_2 * (t+AGE_0)**2 + BETA_w_3 * edu + BETA_w_4 * edu * (t+AGE_0) + BETA_w_5 * edu * (t+AGE_0)**2

# This function generates samples from a normal distribution
e = lambda n: torch.normal(0, SIGMA_e, n)

# Update theta based on previous theta and generated e
theta = lambda theta_p , e: theta_p + e

#wage
wage = lambda u_t, theta_t : torch.maximum(torch.e**(u_t + theta_t) , torch.tensor(w_min))


consumption = lambda a_t, y_t : a_t[:, :-1] + y_t -  a_t[:, 1:]/(1+R)


income_tax  = lambda y_t: y_t - KAPPA * (y_t+1e-8)**(1-TAU)
social_security_tax = lambda y_t: TAU_SS * torch.minimum(y_t, W_b)


Score = lambda y_t, T_S: torch.topk(y_t, T_S).values.mean(dim=-1)


pension_benefit = lambda S: ((BEND_POINTS_b[0] * S * (S <= BEND_POINTS_a[0])) +
                             (BEND_POINTS_b[0] * BEND_POINTS_a[0] + BEND_POINTS_b[1] * (S - BEND_POINTS_a[0])) * ((BEND_POINTS_a[0] < S) & (S <= BEND_POINTS_a[1])) +
                             (BEND_POINTS_b[0] * BEND_POINTS_a[0] + BEND_POINTS_b[1] * (BEND_POINTS_a[1] - BEND_POINTS_a[0]) + BEND_POINTS_b[2] * (S - BEND_POINTS_a[1])) * ((BEND_POINTS_a[1] < S) & (S <= BEND_POINTS_a[2])) +
                             (BEND_POINTS_b[0] * BEND_POINTS_a[0] + BEND_POINTS_b[1] * (BEND_POINTS_a[1] - BEND_POINTS_a[0]) + BEND_POINTS_b[2] * (W_b - BEND_POINTS_a[1])) * (S > BEND_POINTS_a[2]))


def retirement_benefit(all_y, t_R, TS=35):
    delta_t = DELTA_t.to(all_y.device)
    delta = delta_t[t_R]
    S = Score(all_y, TS)
    b = pension_benefit(S) * delta
    
    return b
    