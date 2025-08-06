### Parameters initialization

import torch
import numpy as np

## Life Cycle --------------------

AGE_0 = 22 # Age of model start
AGE_ER = 62 # Age of early retiermnet
AGE_FR = 67 # Age of normal retierment
AGE_LR = 70 # Age of Late Retirement
AGE_D = 82 # Age of Death

T_W = AGE_ER - AGE_0 + 1 # in fixed retierment model: number of years working
T_D = AGE_D - AGE_0 + 1 # period of death
T_R = T_D - T_W + 1

## Initial Distribution --------------------

P_EDU = 0.190156 # Probility of being college educated
A_0 = 0 # initial asset
THETA_0 = 0 # Initial stochastic persistent productivity

## Wage Function --------------------

w_min = 5 # minimum wage
# deterministic wage function parameters
BETA_w_0 = 1.6698 # constant
BETA_w_1 = 0.0605 # Age
BETA_w_2 = -.0006 # Age^2
BETA_w_3 = -.3780 # edu
BETA_w_4 = 0.03214 # edu.Age
BETA_w_5 = -0.0002 # edu.Age^2

SIGMA_e = np.sqrt(.02601) # standard deviation of persistent wage shocks 

## Preference --------------------

BETA = 0.9880 # time predernce
ETA = 0.5 # Fritch elasticity of labor supply
GAMMA = 1.66 # risk aversion
#PSI = 0.01 # coefficient of disutility of hours of work
#PHI = 0.0006 # Coefficent of disutility of labor supply  

## Budget Constaint -------------------- 

cons_min = 2000 # Minimum acceptable consumption level
R = 0.042 # risk-free interset rate

## Grids --------------------

h_max = 2860.0 # Over time work, max possible hours of work
H_FT = 2080.0 # Full time work
H_PT = 1300.0 # Part time work

h_grid = torch.tensor([0.0,H_PT,H_FT,h_max]) # Grids of work hours

## Tax Function --------------------

KAPPA_Inc_Tax = 2.716084 # Income tax constant
TAU_Inc_Tax = 0.1029 # Income tax progressivity

## Social Security Function --------------------

TAU_SS_Tax = 0.106 # Social security tax
Inc_base_SS = torch.tensor(76200) # Wage Base
BEND_POINTS_b_SS = torch.tensor([0.9,0.32,0.15]) # Sopes of pension benefit function
BEND_POINTS_a_SS = torch.tensor([6372,38422,76200]) # Bend points of pension benefit function 
DELTA_t_SS = torch.tensor([0.7, 0.75, 0.8, 0.866, 0.933, 1.00, 1.08, 1.16, 1.24]) # Age of retienetn coefficint of pension  benefit
Len_S_Max_SS = 35 # Number of years count in pension benefit

## Simulation --------------------

#J = 1000000 # Simulation Sample Size


