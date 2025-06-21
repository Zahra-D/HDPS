
from imports import *
from Parameters import *
from Functions import *

sns.set(color_codes=True)


## Histograms --------------------

def Histograms_All(data, edu, var_name, plots_base_dir, epoch):
        
    """
    Plot histograms of the full distribution of a variable (flattened across all ages and individuals),
    separated by education level.
    """
        
    plt.figure(figsize=(15, 7))
    
    # Flatten and plot for edu = 1
    plt.hist(data[edu > 0].view(-1), edgecolor='skyblue', bins=400, alpha=0.4, label='edu = 1')
    # Flatten and plot for edu = 0
    plt.hist(data[edu <= 0].view(-1), edgecolor='orange', bins=400, alpha=0.4, label='edu = 0')
    
    plt.legend()
    plt.title(f'Histogram of all {var_name}')
    plt.xlabel(var_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    # Save figure
    dir_save = f'{plots_base_dir}/epoch{epoch}/histograms'
    pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{dir_save}/hist_{var_name}_all.png')
    plt.close()


def Histograms_Individual_Ages(data, edu, var_name, plots_base_dir=None, epoch=None, save=False):
    
    """
    Plot histograms of the distribution of a variable at selected ages: 25, 40, 55 (and 75 for Asset/Consumption),
    separated by education level.
    """
        
    edu = edu.view(-1)
    show_age_75 = var_name in ['Consumption', 'Asset']
    
    # Prepare figure
    fig, ax = plt.subplots(1, 3 + show_age_75, figsize=(20, 5))
    fig.suptitle(f'Histogram of {var_name} for ages 25, 40, and 55')
    
    # Plot each age
    ages = [25, 40, 55] + ([75] if show_age_75 else [])
    for i, age in enumerate(ages):
        ax[i].hist(data[edu > 0][:, age - AGE_0].view(-1), bins=400, edgecolor='skyblue', alpha=0.2, label='edu = 1')
        ax[i].hist(data[edu <= 0][:, age - AGE_0].view(-1), bins=400, edgecolor='orange', alpha=0.1, label='edu = 0')
        ax[i].set_title(f'Age {age}')
        
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}/histograms'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/hist_{var_name}_25_40_55.png')
        plt.close()
    
    
## Trends --------------------
   

def plot_trend(data, edu, var_name, func, plots_base_dir=None, epoch=None, save=False):
    
    """
    Plot the time trend of a variable (e.g., Asset, Consumption) across ages, separated by education level.
    Args:
        data (Tensor): shape [N, T], variable across individuals and time (e.g., from model.simulate()).
        edu (Tensor): shape [N,], binary education indicator (0 or 1).
        var_name (str): label for the y-axis and file naming (e.g., 'Asset').
        func (str): 'mean' or 'median' for aggregation method.
        plots_base_dir (str): base directory to save plots.
        epoch (int): current epoch for saving path.
        save (bool): whether to save the figure or display it.
    """
    
    edu = edu.view(-1)
    T = data.shape[1]
    ages = list(range(AGE_0, AGE_0 + T))

    plt.figure(figsize=(15, 7))

    if func == 'median':
        # Median across individuals, by age and education group
        plt.plot(ages, data[edu > 0].median(dim=0).values, color='skyblue', label='edu = 1')
        plt.plot(ages, data[edu <= 0].median(dim=0).values, color='orange', label='edu = 0')
    elif func == 'mean':
        # Mean across individuals, by age and education group
        plt.plot(ages, data[edu > 0].mean(dim=0), color='skyblue', label='edu = 1')
        plt.plot(ages, data[edu <= 0].mean(dim=0), color='orange', label='edu = 0')
    else:
        raise ValueError("func must be either 'mean' or 'median'")

    plt.legend()
    plt.title(f'Trend of {var_name}')
    plt.xlabel("Age")
    plt.ylabel(var_name)
    plt.tight_layout()

    if save:
        dir_save = f'{plots_base_dir}/epoch{epoch}/trend'
        pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dir_save}/trend_{var_name}_{func}.png')
        plt.close()
    else:
        plt.show()


## Policy Functions  --------------------
    
# def inverse_wage(wage, t, edu):
    
#     """
#     Computes the implied theta from wage, given time t and education edu.
    
#     Args:
#         wage (Tensor): Wage values (must be > 0)
#         t (Tensor or int): Time index (age - AGE_0), scalar or tensor of same shape as wage
#         edu (Tensor): Binary education indicator (same shape as wage)
    
#     Returns:
#         Tensor: theta values implied by wage
#     """
    
#     # Ensure minimum wage is respected (numerical stability)
#     eps = 1e-6
#     wage = torch.clamp(wage, min=eps)

#     # Broadcast t if scalar
#     if isinstance(t, int):
#         t = torch.full_like(wage, fill_value=t)

#     # Compute deterministic component
#     u = wage_det(edu, t)  # shape-compatible from functions.py

#     # Return implied theta
#     theta = torch.log(wage) - u
#     return theta

# def plot_policy_over_asset(model, edu, all_a, all_w, type='Asset', age_list=[25, 40, 55],
#                            plots_base_dir=None, epoch=None, save=False):
    
#     """
#     Plot model response over asset levels for different wage quantiles and education groups.

#     Args:
#         model: trained Master_Model instance
#         edu: [N,] binary education indicator
#         all_a, all_w: simulated asset and wage paths [N, T]
#         type: 'Asset', 'workhour', or 'Ratio'
#         age_list: list of ages to plot
#         plots_base_dir: base directory to save plots
#         epoch: training epoch
#         save: whether to save the figure
#     """
    
#     device = all_a.device
#     edu = edu.view(-1)
#     fig, ax = plt.subplots(2, len(age_list), figsize=(8 * len(age_list), 10))
#     for i, age in enumerate(age_list):
#         age_idx = age - AGE_0
#         for row, edu_val in enumerate([0, 1]):
#             mask = (edu > 0) if edu_val == 1 else (edu <= 0)
#             sigma = all_a[mask][:, age_idx].std()
#             mean = all_a[mask][:, age_idx].mean()
#             a_grid = torch.linspace(mean - 2 * sigma, mean + 2 * sigma, 1000).to(device)
#             w_q = [all_w[mask][:, age_idx].quantile(q).to(device) for q in [0.25, 0.5, 0.75]]
#             th_q = [inverse_wage(w, age, torch.tensor(edu_val, device=device)) for w in w_q]
#             y_dummy = torch.ones((1000, age_idx), device=device) * 50000
#             def run(theta):
#                 theta_tensor = torch.full((1000, 1), theta.item(), device=device)
#                 edu_tensor = torch.full((1000, 1), edu_val, device=device)
#                 asset_tensor = a_grid.unsqueeze(1).to(device)
#                 y_input = y_dummy.to(device)

#                 return model.working_model(y_input, torch.cat([theta_tensor, edu_tensor, asset_tensor], dim=1), age_idx)

#             curves = []
#             for k, th in enumerate(th_q):
#                 h, x = run(th)
#                 if type == 'workhour':
#                     curve = h
#                     ylabel = 'Work hour t'
#                 else:
#                     wage = w_q[k]
#                     y = h * wage
#                     if type == 'Ratio':
#                         curve = y - social_security_tax(y) - income_tax(y) + a_grid
#                         ylabel = 'After-tax resource ratio'
#                     elif type == 'Asset':
#                         curve = (1 - x.squeeze()) * (y - social_security_tax(y) - income_tax(y) + a_grid) * (1 + R)
#                         ylabel = 'Asset t+1'
#                     else:
#                         raise ValueError("Invalid plot type")
#                 curves.append(curve)
#             for k, curve in enumerate(curves):
#                 ax[row][i].plot(a_grid.detach().cpu(), curve.detach().cpu(), label=f'q={k + 1}')
#             ax[row][i].set_title(f'Age {age}, edu={edu_val}')
#             ax[row][i].set_xlabel('Asset t')
#             ax[row][i].set_ylabel(ylabel)
#             ax[row][i].legend()
#     plt.tight_layout()
#     if save:
#         dir_save = f'{plots_base_dir}/epoch{epoch}/policy'
#         pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
#         plt.savefig(f'{dir_save}/policy_function_{type}_vs_asset.png')
#         plt.close()

    
# def policy_function_plot_wage(model, edu, type, all_a, all_w, plots_base_dir=None, epoch=None, save=False):
    
#     """
#     Plot policy functions over wage values for different asset quantiles and education levels.
#     """
    
#     edu = edu.view(-1)
#     device = all_w.device
#     fig, ax = plt.subplots(2, 3, figsize=(24, 10))
#     for i, age in enumerate([25, 40, 55]):
#         age_idx = age - AGE_0
#         sigma_1 = all_w[edu > 0][:, age_idx].std()
#         sigma_0 = all_w[edu <= 0][:, age_idx].std()
#         mean_1 = all_w[edu > 0][:, age_idx].mean()
#         mean_0 = all_w[edu <= 0][:, age_idx].mean()
#         a_q_edu1 = [all_a[edu > 0][:, age_idx].quantile(q).to(device) for q in [0.25, 0.5, 0.75]]
#         a_q_edu0 = [all_a[edu <= 0][:, age_idx].quantile(q).to(device) for q in [0.25, 0.5, 0.75]]
#         def safe_range(mean, sigma):
#             start = max(w_min, mean - 2 * sigma)
#             end = mean + 2 * sigma
#             return torch.linspace(start, end, 1000).to(device)
#         w_vals_1 = safe_range(mean_1, sigma_1)
#         w_vals_0 = safe_range(mean_0, sigma_0)
#         th_vals_1 = inverse_wage(w_vals_1, age, torch.ones_like(w_vals_1, device=device))
#         th_vals_0 = inverse_wage(w_vals_0, age, torch.zeros_like(w_vals_0, device=device))
#         y_dummy = torch.ones((1000, age_idx), device=device) * 50000
#         def run_block(th, edu_val, asset_val):
#             return model.working_model(
#                 y_dummy,
#                 torch.stack([th, torch.full_like(th, edu_val), torch.full_like(th, asset_val)], dim=1),
#                 age_idx
#             )
#         curves_edu1 = [run_block(th_vals_1, 1, a_val) for a_val in a_q_edu1]
#         curves_edu0 = [run_block(th_vals_0, 0, a_val) for a_val in a_q_edu0]
#         def compute_curves(h_list, x_list, w_vals, a_q_list):
#             y_list = [h * w_vals for h in h_list]
#             if type == 'workhour':
#                 return h_list, 'Work hours t'
#             elif type == 'Ratio':
#                 curves = [(y - social_security_tax(y) - income_tax(y) + a_q) for y, a_q in zip(y_list, a_q_list)]
#                 return curves, 'After-tax resource ratio'
#             elif type == 'Asset':
#                 curves = [
#                     (1.0 - x.squeeze()) * (y - social_security_tax(y) - income_tax(y) + a_q) * (1 + R)
#                     for y, x, a_q in zip(y_list, x_list, a_q_list)
#                 ]
#                 return curves, 'Asset t+1'
#             else:
#                 raise ValueError("Invalid plot type")
#         h_1, x_1 = zip(*curves_edu1)
#         h_0, x_0 = zip(*curves_edu0)
#         curves_1, ylabel = compute_curves(h_1, x_1, w_vals_1, a_q_edu1)
#         curves_0, _ = compute_curves(h_0, x_0, w_vals_0, a_q_edu0)
#         for k, curve in enumerate(curves_1):
#             ax[1][i].plot(w_vals_1.cpu(), curve.detach().cpu(), label=f'qa={a_q_edu1[k]:.2f}')
#         ax[1][i].set_title(f'Age {age}, edu=1')
#         ax[1][i].set_xlabel("Wage t")
#         ax[1][i].set_ylabel(ylabel)
#         ax[1][i].legend()
#         for k, curve in enumerate(curves_0):
#             ax[0][i].plot(w_vals_0.cpu(), curve.detach().cpu(), label=f'qa={a_q_edu0[k]:.2f}')
#         ax[0][i].set_title(f'Age {age}, edu=0')
#         ax[0][i].set_xlabel("Wage t")
#         ax[0][i].set_ylabel(ylabel)
#         ax[0][i].legend()
#     plt.tight_layout()
#     if save:
#         dir_save = f'{plots_base_dir}/epoch{epoch}/policy'
#         pathlib.Path(dir_save).mkdir(parents=True, exist_ok=True)
#         fig.savefig(f'{dir_save}/policy_function_{type}_vs_wage.png')
#         plt.close()