#Parameters initialization
import torch
import numpy as np

T_FR = 67
T_ER = 62
T_LR = 70 
T_D = 82

T = 40
J = 1000000

AGE_0 = 22

P_EDU = 0.190156

w_min = 5
consumption_min = 2000


BETA_w_0 = 1.6698
BETA_w_1 = 0.0605
BETA_w_2 = -.0006
BETA_w_3 = -.3780
BETA_w_4 = 0.03214
BETA_w_5 = -0.0002

SIGMA_e = np.sqrt(.02601)

THETA_0 = 0

# PSI = 0.04

PSI_R = 0.01
T_R = 10

BETA = 0.98803
#generating all the beta_t, BETA_t: [41]



ETA = 0.5
GAMMA = 1.66

R = 0.042


# asset for the first year
A_1 = 0



#the defined work hours
H = torch.tensor([0.0,1300.0,2080.0,2860.0])


KAPPA = 2.716084
TAU = 0.1029

TAU_SS = 0.106
W_b = torch.tensor(76200) 


PHI = 0.0006

T_S = 35

BEND_POINTS_b = torch.tensor([0.9,0.32,0.15])
BEND_POINTS_a = torch.tensor([6372,38422,76200])




DELTA_t = torch.tensor([0.7, 0.75, 0.8, 0.866, 0.933, 1.00, 1.08, 1.16, 1.24])