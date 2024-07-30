import torch
class Economic:
    
    T_FR = 67
    T_ER = 62
    T_LR = 70 
    T_D = 82
    
    AGE_0 = 22
    A_1 = 0.0
    
    #generating samples
    P_EDU = 0.190156
    J = 1000000
    
    SIGMA_e = torch.sqrt(torch.tensor(.02601))
    THETA_0 = 0
    P_EDU = 0.190156
    minimum_wage = 5.0
    
    #mu
    BETA_w_0 = 1.6698
    BETA_w_1 = 0.0605
    BETA_w_2 = -.0006
    BETA_w_3 = -.3780
    BETA_w_4 = 0.03214
    BETA_w_5 = -0.0002
    
    #asset
    R = 0.042
    

    

    
    
    # income_tax
    KAPPA = 2.716084
    TAU = 0.1029


    #social_security_tax
    TAU_SS = 0.106
    W_b = torch.tensor(76200) 
    
    #retimerment
    T_S = 35
    BEND_POINTS_b = torch.tensor([0.9,0.32,0.15])
    BEND_POINTS_a = torch.tensor([6372,38422,76200])
    DELTA_t = torch.tensor([0.7, 0.75, 0.8, 0.866, 0.933, 1.00, 1.08, 1.16, 1.24])
    
    
    
    H = torch.tensor([0.0,1300.0,2080.0,2860.0])
    




   
    
    #utitlity
    GAMMA = 1.66
    BETA = 0.98803
    ETA = 0.5
    
    
    
    
    
    #mean of work hour for year 55
    M_D1 = 2200
    #percentage of 60-year-old people who still works
    M_D2 = .75
    
    
    
    
    @staticmethod
    # This function generates samples from a normal distribution given number of samples
    def e(n):
        return torch.normal(0, Economic.SIGMA_e, n)
    
    @staticmethod
    # Update theta based on previous theta and generated e
    def theta(theta_p, e):    
        return theta_p + e
    
    @staticmethod
    def mu(edu, t):
        return Economic.BETA_w_0 + Economic.BETA_w_1 * (t) + Economic.BETA_w_2 * (t)**2 + Economic.BETA_w_3 * edu + Economic.BETA_w_4 * edu * (t) + Economic.BETA_w_5 * edu * (t)**2
        
    @staticmethod
    def wage(mu, theta):
        return  torch.maximum(torch.e**(mu + theta) , torch.tensor(Economic.minimum_wage))
    
    
    
    @staticmethod
    def consumption_asset_cashInHand(consumption_fraction, income, current_asset, type = 'working'):
        
        if type== 'working':
            cash_in_hand = income - Economic.income_tax(income) -Economic.social_security_tax(income) + current_asset
        elif type== 'retired':
            cash_in_hand =  income + current_asset
        
        consumption = consumption_fraction * cash_in_hand + 1e-8
        next_year_asset = (1.0 - consumption_fraction) * cash_in_hand * (1+Economic.R) 
        
        return consumption, next_year_asset, cash_in_hand
        
    
    
    
    
    
    
    
    @staticmethod
    def income_tax(income):
        return  income - Economic.KAPPA * (income+1e-8)**(1-Economic.TAU)
    
    @staticmethod
    def social_security_tax(income):
        return Economic.TAU_SS * torch.minimum(income, Economic.W_b)
       
       
        
        
    @staticmethod
    def retirement_benefit(all_y, t_R, TS=None):

        if TS is None:
            TS = Economic.T_S    
        delta_t = Economic.DELTA_t.to(all_y.device)
        delta = delta_t[t_R]
        score = torch.topk(all_y, TS).values.mean(dim=-1)
        
        pension_benefit = ((Economic.BEND_POINTS_b[0] * score * (score <= Economic.BEND_POINTS_a[0])) +
                             (Economic.BEND_POINTS_b[0] * Economic.BEND_POINTS_a[0] + Economic.BEND_POINTS_b[1] * (score - Economic.BEND_POINTS_a[0])) * ((Economic.BEND_POINTS_a[0] < score) & (score <= Economic.BEND_POINTS_a[1])) +
                             (Economic.BEND_POINTS_b[0] * Economic.BEND_POINTS_a[0] + Economic.BEND_POINTS_b[1] * (Economic.BEND_POINTS_a[1] - Economic.BEND_POINTS_a[0]) + Economic.BEND_POINTS_b[2] * (score - Economic.BEND_POINTS_a[1])) * ((Economic.BEND_POINTS_a[1] < score) & (score <= Economic.BEND_POINTS_a[2])) +
                             (Economic.BEND_POINTS_b[0] * Economic.BEND_POINTS_a[0] + Economic.BEND_POINTS_b[1] * (Economic.BEND_POINTS_a[1] - Economic.BEND_POINTS_a[0]) + Economic.BEND_POINTS_b[2] * (Economic.W_b - Economic.BEND_POINTS_a[1])) * (score > Economic.BEND_POINTS_a[2]))
        b = pension_benefit * delta
        
        return b
    
    
    
    
    
    
    
    
    
    @staticmethod
    def utility(c_t, h_t , phi, psi, BETA_t):
        consumption_utility =  (c_t**(1-Economic.GAMMA))/(1-Economic.GAMMA)
        work_hour_disutility = ((h_t/Economic.H[-1]) ** (1+1/Economic.ETA))/(1+1/Economic.ETA)
        working_disutility =  (h_t > 50).int()
        utility = (BETA_t * (consumption_utility - phi * working_disutility - psi * work_hour_disutility))
        return utility

  
        