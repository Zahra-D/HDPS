from source.utils.imports import * 
from .working_block import WorkYearBlock
from .retirement_block import RetirementYearBlock
from source.economic import Economic
# from Parameters import AGE_0, T_ER, R
# from functions import income_tax, social_security_tax, retirement_benefit


class EarlyRetiermentBlock(nn.Module):
  
      def __init__(self, year=61,num_hidden_node_w = 10, num_hidden_node_r = 5, alpha_pr = 1, hard_gumbel = True, layers_dict=None):
        super().__init__()
        
        assert (year >= 62) and (year<=70)
        
        self.working_block = WorkYearBlock(num_input=year - Economic.AGE_0 + 3, num_hidden_node = num_hidden_node_w,  mode= 'early_retirement_year', alpha_pr=alpha_pr,  hard_gumbel=hard_gumbel, layers_dict=layers_dict)
        self.retirement_block = RetirementYearBlock(num_hidden_unit=num_hidden_node_r, year=year)
        
        self.year= year

        
        
        
        
        
        
        

      def forward(self, theta, edu, a_w_t, a_r_t, all_y, w_t, pr_bar_t, b_bar_t):
        
        h_t, x_ww, pr_t, x_rw = self.working_block(theta, edu, a_w_t, all_y) 
        # t = torch.ones_like(a_r_t).to(a_r_t.device) * self.year
        x_rr = self.retirement_block(a_r_t, b_bar_t)
          
          

        y_t = w_t * h_t 
        
      
  
        
        #calculating a_ww a_rw a_rr
        
        c_ww_t, a_ww_tp, _ = Economic.consumption_asset_cashInHand(x_ww, y_t, a_w_t, type='working')
        # c_ww_t = x_ww * (y_t - income_tax(y_t) -social_security_tax(y_t) + a_w_t) + 1e-8
        # a_ww_tp = (1.0 - x_ww)*((y_t) - income_tax(y_t) - social_security_tax(y_t) + a_w_t)* (1+R) 
        
        b_t = Economic.retirement_benefit(all_y, self.year - Economic.T_ER, Economic.T_S)
        
        
        c_rw_t, a_rw_tp, _ = Economic.consumption_asset_cashInHand(x_rw, b_t, a_w_t, type='retired')
        # c_rw_t = (x_rw *(b_t + a_w_t))+ 1e-8
        # a_rw_tp = ((1.0-x_rw)*(b_t + a_w_t)*(1+R))


        c_rr_t, a_rr_tp, _ = Economic.consumption_asset_cashInHand(x_rr, b_bar_t, a_r_t, type='retired')
        # a_rr_tp = ((1.0-x_rr)*(b_bar_t + a_r_t)*(1+R))
        # c_rr_t = (x_rr *(b_bar_t + a_r_t))+ 1e-8
        
        #the asset for next year, the a_w_tp is the same as a_ww_t so do not use a new varialbe for it
        a_r_tp = (1-pr_bar_t) * ( a_rw_tp )  +  pr_bar_t*(a_rr_tp)
        
        
        
        #update pr_bar
        pr_bar_tp  = pr_bar_t + (1-pr_bar_t) * pr_t[:,0]
        
        #calculate a_tp
        a_tp = (1-pr_bar_tp) * a_ww_tp +  pr_bar_tp*(a_r_tp)
        
        #update b_bar
        b_bar_tp = pr_bar_t * b_bar_t + (1-pr_bar_t) * b_t
        

        
        return {'a_w_tp':a_ww_tp,
                'a_r_tp': a_r_tp,
                
                
                
                # 'c_t':c_t,
                'a_tp': a_tp,
                
                
                # 'a_ww_tp': a_ww_tp,
                'c_ww_t': c_ww_t,
                
                # 'a_rw_tp': a_rw_tp,
                'c_rw_t': c_rw_t,                

                # 'a_rr_tp': a_rr_tp,
                'c_rr_t': c_rr_t,
                
                'y_t': y_t,
                'h_t': h_t,
                'pr_t': pr_t[:,0],
                'pr_bar_tp': pr_bar_tp, 

                'b_bar_tp': b_bar_tp}
        
      