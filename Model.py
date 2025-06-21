
import torch
import torch.nn as nn
import torch.nn.functional as F

from Parameters import *
from Functions import *


### === Working Model === ###


class Working_Model(nn.Module):  # Defines a neural network model for working years of the agent

    # DNN Architecture --------------------

    def __init__(self, SS_Type, SS_Param=None, d_model=64, num_heads=4, dropout=0.0, y_gate_temp=10.0):  # Initialize model with architecture and economic parameters
        super().__init__()  # Call the parent constructor from nn.Module
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"  # Ensure attention heads evenly divide d_model

        self.SS_Type = SS_Type  # Store structural summary type (e.g., Life_Time, Top_Years, etc.)
        self.SS_Param = SS_Param  # Parameter specific to SS_Type (e.g., K for Top_Years, L for Last_Years)
        self.d_model = d_model  # Hidden dimension used throughout the model
        self.num_heads = num_heads  # Number of attention heads
        self.y_gate_temp = y_gate_temp  # Temperature used in soft thresholding (Top_Years)

        self.income_proj = nn.Linear(3, d_model) if SS_Type in ['Non_Parametric', 'Top_Years'] else None  # If attention-based, project income + position + rank into embedding space
        self.static_proj = nn.Linear(1, d_model)  # Project scalar static inputs (theta, edu, a) into embedding space

        if SS_Type == 'Non_Parametric':  # If fully non-parametric attention architecture
            self.attn_layers = nn.ModuleList([  # Build two layers of attention
                nn.LayerNorm(d_model),  # Normalize inputs before attention
                nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True),  # First attention layer
                nn.ReLU(),  # Activation after first attention
                nn.LayerNorm(d_model),  # Normalize again
                nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True),  # Second attention layer
                nn.ReLU()  # Activation after second attention
            ])
        elif SS_Type == 'Top_Years':  # If using Top-K attention
            self.attn_layers = nn.ModuleList([
                nn.LayerNorm(d_model),  # Normalize before attention
                nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True),  # Single attention layer
                nn.ReLU()  # Activation after attention
            ])
        else:
            self.attn_layers = nn.ModuleList([])  # Safe fallback for other SS_Type

        self.general_layers = nn.Sequential(  # Feed-forward MLP block for simpler summary types (Life_Time, Last_Years)
            nn.BatchNorm1d(4),  # Normalize 3-element input: [theta, edu, a and mean_income]
            nn.Linear(4, d_model),  # Project input to hidden space
            nn.ReLU(),  # Activation
            # nn.BatchNorm1d(d_model),  # Normalize
            # nn.Linear(d_model, d_model),  # Hidden layer
            # nn.ReLU(),  # Activation
            # nn.ReLU(),  # Activation
            # nn.BatchNorm1d(d_model),  # Normalize
            # nn.Linear(d_model, d_model),  # Hidden layer
            # nn.ReLU(),  # Activation
            nn.BatchNorm1d(d_model),  # Normalize
            nn.Linear(d_model, d_model),  # Output projection
            nn.ReLU()  # Activation
        )

        self.task_layer_h_core = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        self.task_layer_x_core = nn.Sequential(  # Shared task head for x (consumption share)
            nn.Linear(d_model, d_model),  # Hidden layer
            nn.GELU(),  # Activation
            nn.Linear(d_model, 1),  # Output scalar
            #nn.Sigmoid()  # Between 0 and 1
        )

        self.task_layer_h_residual = nn.ModuleList([
            nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
            ) for _ in range(T_W)
        ])

        self.task_layer_x_residual = nn.ModuleList([  # Year-specific residual heads for x
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                #nn.Sigmoid()
            ) for _ in range(T_W)
        ])

        # Initialization --------------------

        # General Xavier init for all Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # BatchNorms: standard identity init
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Attention Projections (MultiheadAttention)
        for m in self.attn_layers:
            if isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_uniform_(m.out_proj.weight)
                nn.init.zeros_(m.out_proj.bias)

        # income_proj (projects [income, position, rank])
        if self.income_proj is not None:
            nn.init.xavier_uniform_(self.income_proj.weight)
            nn.init.zeros_(self.income_proj.bias)

        # static_proj (projects [theta, edu, a])
        nn.init.xavier_uniform_(self.static_proj.weight, gain=0.5)
        nn.init.zeros_(self.static_proj.bias)

        # Core task head biases for hours (target ~1800 hours) → sigmoid(0.63) ≈ 1800 / 2860
        nn.init.constant_(self.task_layer_h_core[-1].bias, torch.logit(torch.tensor(0.063)))

        # Core task head biases for consumption share (target ~50% for workers)
        nn.init.constant_(self.task_layer_x_core[-1].bias, torch.logit(torch.tensor(0.05)))

        # Residual task head biases (start neutral)
        for head in self.task_layer_h_residual:
            nn.init.zeros_(head[-1].bias)

        for head in self.task_layer_x_residual:
            nn.init.zeros_(head[-1].bias)


    def soft_rank(self, y_seq, tau=1.0):  # Differentiable soft rank function
        B, T = y_seq.shape  # Batch size and time length
        y_i = y_seq.unsqueeze(2)  # (B, T, 1)
        y_j = y_seq.unsqueeze(1)  # (B, 1, T)
        diff = (y_j - y_i) / tau  # Pairwise differences scaled by temperature
        P = torch.sigmoid(diff)  # Approximate rank by sigmoid comparisons
        return P.sum(dim=2) / T  # Average ranks

    # Forward Pass  --------------------

    def forward(self, y_seq, static_inputs, year):  # Forward pass: compute h and x
        B, T = y_seq.shape  # Get batch and time dims

        if self.SS_Type == 'Life_Time':  # If using average of income
            mean_income = y_seq.mean(dim=1, keepdim=True)
            summary = torch.cat([static_inputs, mean_income], dim=1)  # Append mean_income to static features
            summary = self.general_layers(summary)  # Pass through MLP

        elif self.SS_Type == 'Top_Years':  # If using top-K income years
            k = self.SS_Param  # Number of top years
            rank = self.soft_rank(y_seq)  # Compute soft rank
            threshold = k / T  # Normalize threshold
            gate = torch.sigmoid((rank - threshold) * self.y_gate_temp).to(y_seq.device)  # Gate signal based on threshold
            position = torch.arange(T, device=y_seq.device).float().unsqueeze(0).expand(B, T) / T  # Time position
            y_features = torch.stack([y_seq, position, rank], dim=-1).to(y_seq.device)  # Stack features
            y_proj = self.income_proj(y_features) * gate.to(y_seq.device).unsqueeze(-1)  # Masked projection
            static_proj = self.static_proj(static_inputs.to(y_seq.device).unsqueeze(-1))  # Static tokens
            all_tokens = torch.cat([y_proj, static_proj], dim=1)  # Concat for attention
            seq_mask = (y_seq != 0).float()  # Mask for valid income
            static_mask = torch.ones(B, 3, device=y_seq.device)  # All static tokens valid
            attn_mask = torch.cat([seq_mask, static_mask], dim=1)  # Combine masks
            padding_mask = attn_mask == 0  # Create binary mask
            x = self.attn_layers[0](all_tokens)  # Normalize
            attn_out, _ = self.attn_layers[1](x, x, x, key_padding_mask=padding_mask)  # Attention
            attn_out = self.attn_layers[2](attn_out)  # Activation
            summary = attn_out[:, -1, :]  # Use final token as summary

        elif self.SS_Type == 'Last_Years':  # If using last L years of income
            L = self.SS_Param
            if year >= T_W - L + 2:
                start = max(0, T_W - L + 1)
                end = year
                mean_past = y_seq[:, start:end].mean(dim=1, keepdim=True)
                x = torch.cat([static_inputs, mean_past], dim=1)
            else:
                x = static_inputs
            summary = self.general_layers(x)


        elif self.SS_Type == 'Non_Parametric':  # Fully flexible attention-based model
            position = torch.arange(T, device=y_seq.device).float().unsqueeze(0).expand(B, T) / T  # Add time position
            rank = self.soft_rank(y_seq)  # Soft rank
            y_features = torch.stack([y_seq, position, rank], dim=-1).to(y_seq.device)  # Full feature set
            y_proj = self.income_proj(y_features)  # Project
            static_proj = self.static_proj(static_inputs.to(y_seq.device).unsqueeze(-1))  # Project static
            all_tokens = torch.cat([y_proj, static_proj], dim=1)  # Combine
            seq_mask = (y_seq != 0).float()  # Valid tokens
            static_mask = torch.ones(B, 3, device=y_seq.device)  # Create a mask of ones for static tokens (always active)
            attn_mask = torch.cat([seq_mask, static_mask], dim=1)  # Concatenate sequence and static masks for attention
            padding_mask = attn_mask == 0  # Convert attention mask to padding mask (True for invalid/padded tokens)
            x = self.attn_layers[0](all_tokens)  # Apply first LayerNorm to all tokens
            attn_out, _ = self.attn_layers[1](x, x, x, key_padding_mask=padding_mask)  # Apply first Multihead Attention using the padding mask
            attn_out = self.attn_layers[2](attn_out)  # Apply ReLU activation to attention output
            x = self.attn_layers[3](attn_out)  # Apply second LayerNorm to prepare for next attention
            attn_out, _ = self.attn_layers[4](x, x, x, key_padding_mask=padding_mask)  # Apply second Multihead Attention
            attn_out = self.attn_layers[5](attn_out)  # Apply ReLU activation to final attention output
            summary = attn_out[:, -1, :]  # Final summary token

        else:
            raise ValueError(f"Unsupported SS_Type: {self.SS_Type}")  # Catch invalid configuration

        h_core = self.task_layer_h_core(summary)  # Predict core h
        h_res = self.task_layer_h_residual[year](summary)  # Year-specific h residual
        h_sum = h_core + h_res
        h_nor = (h_sum - h_sum.mean()) / (h_sum.std() + 1e-5)
        # Sigmoid last layer model
        h_out = torch.sigmoid(h_nor)  # Total h
        # Linear last layer model
        #h_out = torch.clamp(h_core + h_res, min=1e-3, max=1.000)
        # No labor Supply
        #h_out = torch.full((static_inputs.shape[0], 1), H_FT/h_max, device=static_inputs.device)

        x_core = self.task_layer_x_core(summary)  # Predict core x
        x_res = self.task_layer_x_residual[year](summary)  # Year-specific x residual
        x_sum = x_core + x_res
        x_nor = (x_sum - x_sum.mean()) / (x_sum.std() + 1e-5)
        # Sigmoid last layer model
        x_out = torch.sigmoid(x_nor)  # Total x
        x_out = torch.clamp(x_out, 1e-3) # No zero consumption
        # Linear last layer model
        # x_out = torch.clamp(x_core + x_res, min=1e-3, max=1.000)

        return h_out, x_out  # Return decisions


### === Retiree Model === ###


class Retiree_Model(nn.Module):  # Model that simulates retirement behavior based on savings and pension

    # DNN Architecture --------------------

    def __init__(self, num_hidden_units_R=5, activation_function=nn.GELU):  # Constructor: initializes model components
        super().__init__()  # Call parent constructor from nn.Module

        # DNN: Highway Model (Core plus Age-Specific Residual Taks Layers)

        self.gen = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, num_hidden_units_R),
            nn.GELU()
        )

        self.core = nn.Sequential(
            #nn.BatchNorm1d(num_hidden_units_R),
            nn.Linear(num_hidden_units_R, num_hidden_units_R),
            nn.GELU(),
            nn.Linear(num_hidden_units_R, 1)
        )

        self.res = nn.ModuleList([
            nn.Sequential(
                #nn.BatchNorm1d(num_hidden_units_R),
                nn.Linear(num_hidden_units_R, num_hidden_units_R),
                nn.GELU(),
                nn.Linear(num_hidden_units_R, 1)
            ) for _ in range(T_R)
        ])

        # DNN: MLP with Age as Input
        # self.model = nn.Sequential(  # Define model layers in sequence
        #     nn.BatchNorm1d(3),  # Normalize input: year, asset, benefit
        #     nn.Linear(3, num_hidden_units_R),  # Project input to hidden space
        #     activation_function(),  # Apply non-linearity (e.g., GELU)
        #     # nn.BatchNorm1d(num_hidden_units_R),  # Normalize input: year, asset, benefit
        #     # nn.Linear(num_hidden_units_R, num_hidden_units_R),  # Project input to hidden space
        #     # activation_function(),  # Apply non-linearity (e.g., GELU)
        #     nn.BatchNorm1d(num_hidden_units_R),  # Normalize input: year, asset, benefit
        #     nn.Linear(num_hidden_units_R, 1),  # Project to scalar
        #     nn.Sigmoid()  # Output in [0, 1] for consumption share
        # )


        # Initialization  --------------------

        # DNN: MLP with Age as Input
        # for m in self.modules():  # Loop over all modules in the model
        #     if isinstance(m, nn.Linear):  # Check if module is a Linear layer
        #         nn.init.xavier_uniform_(m.weight)  # Initialize weights with Xavier uniform distribution
        #         if m.bias is not None:  # If bias exists
        #             nn.init.zeros_(m.bias)  # Initialize biases to zero

        # nn.init.constant_(self.model[-2].bias, torch.logit(torch.tensor(0.05))) # RETIREE CONSUMPTION TO 50 %
        
        # DNN: Highway Model (Core plus Age-Specific Residual Taks Layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.constant_(self.core[-1].bias, torch.logit(torch.tensor(0.05)))
        for res in self.res:
            nn.init.zeros_(res[-1].bias)

    # Forward Pass  --------------------

    # DNN: MLP with Age as Input
    #def forward(self, year, a_t, b_t): 
        # x = torch.cat([year.unsqueeze(1), a_t.unsqueeze(1), b_t.unsqueeze(1)], dim=1)  # Stack year, asset, and benefit as input features
        # x = self.model(x)  # [B, 1]
        # x = torch.clamp(x, min= 1e-3)  # Clamp post-sigmoid, no zero consumption
        # return x  # Return predicted x (consumption share)

    # DNN: Highway Model (Core plus Age-Specific Residual Taks Layers)
    def forward(self, a_t, b_t, t):  # t ∈ [0, T_R-1]
        x_in = torch.cat([a_t.unsqueeze(1), b_t.unsqueeze(1)], dim=1)  # [B, 2]
        x_gen = self.gen(x_in)                                         # [B, hidden]
        x_core = self.core(x_gen)                                      # [B, 1]
        x_res = self.res[t](x_gen)                                     # [B, 1]
        x = torch.sigmoid(x_core + x_res)                              # [B, 1]
        return torch.clamp(x, min=1e-3)


### === Master Model === ###


class Master_Model(nn.Module):  # Combines Working_Model and Retiree_Model for full life-cycle simulation

    # DNN Architecture --------------------

    def __init__(self, working_model: Working_Model, retiree_model: Retiree_Model):  # Constructor: initializes model components
        super().__init__()  # Call parent constructor from nn.Module
        self.working_model = working_model  # NN for working years
        self.retiree_model = retiree_model  # NN for retirement years

    # Life Cycle Simulatin  --------------------

    def simulate(self, theta_sim_t, edu_sim, a_sim_0):  # Run simulation of lifecycle using working and retirement models
        B = theta_sim_t.shape[0]  # Batch size 
        device = theta_sim_t.device  # Device where tensors are stored

        # Initialize matrices
        w_sim_t = []  # Wages
        y_sim_t = []  # Labor income
        tax_sim_t = [] # All taxes
        re_sim_t = []  # Resources available
        x_sim_t = []  # Consumption share
        h_sim_t = []  # Hours worked
        c_sim_t = []  # Consumption
        a_list = [a_sim_0]  # Set initial assets

        # Working years simulation

        for t in range(T_W):  # Loop over working years
            a_prev = a_list[-1]
            
            if t == 0:
                y_input = torch.zeros(B, 1, device=device)  # No past income in first year
            else:
                y_input = torch.stack(y_sim_t, dim=1) # Use past income as input
                
            static_inputs = torch.stack([theta_sim_t[:, t], edu_sim, a_prev], dim=1).to(device)  # Stack static inputs
            h_out, x_out = self.working_model(y_input, static_inputs, t)  # Predict h and x using working model

            mu = wage_det(edu_sim, t)  # determisitic part of wage
            w_t = wage(mu, theta_sim_t[:, t])  # Apply the wage function
            h_t = h_out.squeeze() * h_max  # Compute hours worked
            x_t = x_out.squeeze()  # Store x output

            y_t = w_t * h_t  # Compute income
            tax_t = income_tax(y_t) + social_security_tax(y_t)
            re_t = y_t - tax_t + a_prev # Compute net resources
            c_t = x_t * re_t  # Compute consumption
            a_next = (1 - x_t) * (1 + R) * re_t # Compute next asset
            # Append to lists
            w_sim_t.append(w_t)
            h_sim_t.append(h_t)
            y_sim_t.append(y_t)
            tax_sim_t.append(tax_t)
            re_sim_t.append(re_t)
            c_sim_t.append(c_t)
            x_sim_t.append(x_t)
            a_list.append(a_next)

        #b_sim = retirement_benefit(torch.stack(y_sim_t, dim=1), 6, self.working_model.SS_Param) # Compute fixed benefit for retirement
        b_sim = retirement_benefit(torch.stack(y_sim_t, dim=1), t_R = 6, SS_Type = self.working_model.SS_Type, SS_Param = self.working_model.SS_Param)

        # Retirement years simulation

        for t in range(T_W, T_D + 1):  # Loop over retirement years
            t_R_index = t - T_W

            a_prev = a_list[-1]
            h_t = torch.zeros(B, device=device) # No work during retierment
            # DNN: Highway Model (Core plus Age-Specific Residual Taks Layers)
            #x_t = self.retiree_model(torch.full((B,), t, dtype=torch.float32, device=device), a_prev, b_sim).squeeze()  # Predict x using retiree model
            # DNN: Highway Model (Core plus Age-Specific Residual Taks Layers)
            x_t = self.retiree_model(a_prev, b_sim, t_R_index).squeeze()
            re_t = b_sim + a_prev  # Compute available resources
            c_t = x_t * re_t  # Compute consumption
            a_next = (1 - x_t) * (1 + R) * re_t # Compute next asset
            # Append to lists
            h_sim_t.append(h_t)
            x_sim_t.append(x_t)
            re_sim_t.append(re_t)
            c_sim_t.append(c_t)
            a_list.append(a_next)
            
        # Stack all time-series outputs

        return {
            'h_sim_t': torch.stack(h_sim_t, dim=1),
            'x_sim_t': torch.stack(x_sim_t, dim=1),
            'w_sim_t': torch.stack(w_sim_t, dim=1),
            'y_sim_t': torch.stack(y_sim_t, dim=1),
            'tax_sim_t': torch.stack(tax_sim_t, dim=1),
            'c_sim_t': torch.stack(c_sim_t, dim=1),
            're_sim_t': torch.stack(re_sim_t, dim=1),
            'a_sim_t': torch.stack(a_list, dim=1),
            'b_sim': b_sim
    }
