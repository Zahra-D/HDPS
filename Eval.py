
from Parameters import *
from Functions import *
from Plots import *
from Model import *
from torch.utils.data import DataLoader
import torch
import torch.serialization
import pathlib
import json
import argparse
import numpy as np
import random
import pandas as pd
import os

print("Starting Evaluation...")

## Configuration --------------------

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_title", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--num_sim", type=int, default=100000)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=100000)
parser.add_argument("--seed_eval", type=int, default=776)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

# Paths
save_dir = pathlib.Path(args.save_dir)
base_dir = save_dir / args.experiment_title

model_path = base_dir / f"model/epoch{args.num_epochs}/model.pt"
plot_dir = base_dir / "plot"
Sim_Results_dir = base_dir / "Sim_Results"
Sim_Results_dir.mkdir(parents=True, exist_ok=True)

# Load Model 
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
epoch = args.num_epochs # Final epoch from Numerical_Exercise.py to use

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

# torch.serialization.add_safe_globals({
#     'Master_Model': Master_Model,
#     'Working_Model': Working_Model,
#     'Retiree_Model': Retiree_Model
# })

model = torch.load(model_path, map_location=device, weights_only=False)
print(vars(model.training_args))
model.to(device)
model.eval() # Switch to evaluation mode 


## Evaluation --------------------


# Set random seeds of Evaulation Dataset
torch.manual_seed(args.seed_eval)
np.random.seed(args.seed_eval)
random.seed(args.seed_eval)

# Generate Evaluation Dataset
dataset_eval = generating_dataset(args.num_sim, T_W, THETA_0, P_EDU)
dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)

# Evaluation Function

def do_eval(model, dataloader, device):
    all_edu, all_theta, all_w, all_h, all_y, all_tax, all_re, all_c, all_a, all_b  = [], [], [], [], [], [], [], [], [], []

    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            theta_t, w_t, edu = batch
            theta_t, w_t, edu = theta_t.to(device), w_t.to(device), edu.to(device)
            a_0 = torch.tensor([A_0] * len(theta_t)).to(device)

            out = model.simulate(theta_t, edu, a_0)
            all_edu.append(edu.cpu())
            all_theta.append(theta_t.cpu())
            all_w.append(out["w_sim_t"].cpu())
            all_h.append(out["h_sim_t"].cpu())
            all_y.append(out["y_sim_t"].cpu())
            all_tax.append(out["tax_sim_t"].cpu())
            all_re.append(out["re_sim_t"].cpu())
            all_c.append(out["c_sim_t"].cpu())
            all_a.append(out["a_sim_t"][:, :-1].cpu())
            all_b.append(out["b_sim"].cpu())

    return [torch.cat(tensors, dim=0) for tensors in [all_edu, all_theta, all_w, all_h, all_y, all_tax, all_re, all_c, all_a, all_b]]

# Run Evaluation

all_edu, all_theta, all_w, all_h, all_y, all_tax, all_re, all_c, all_a, all_b = do_eval(model, dataloader_eval, device)


## Export Results --------------------

# Pack the results into a dict
result_tensors = {
    "education": all_edu.unsqueeze(1),  # Make 2D for CSV
    "persistent_wage_shock": all_theta,
    "wage": all_w,
    "hours_worked": all_h,
    "income": all_y,
    "taxes": all_tax,
    "resource": all_re,
    "consumption": all_c,
    "asset": all_a,
    "pension_benefit": all_b.unsqueeze(1),
}

# Save each tensor to CSV
for name, tensor in result_tensors.items():
    df = pd.DataFrame(tensor.cpu().numpy())
    df.to_csv(Sim_Results_dir / f"{name}.csv", index=False)

## Generate and Save Plots --------------------

print("Saving plots to:", plot_dir)

for var, name in zip([all_a, all_c, all_h, all_y], ['Asset', 'Consumption', 'Workhour', 'Income']):
    Histograms_Individual_Ages(var, all_edu, name, plots_base_dir=plot_dir, epoch=epoch, save=True)
    plot_trend(var, all_edu, name, func='median', plots_base_dir=plot_dir, epoch=epoch, save=True)
    plot_trend(var, all_edu, name, func='mean', plots_base_dir=plot_dir, epoch=epoch, save=True)

# for kind in ['workhour', 'Asset', 'Ratio']:
#     plot_policy_over_asset(model, all_edu, all_a, all_w, type=kind, plots_base_dir=plot_dir, epoch=epoch, save=True)
#     policy_function_plot_wage(model, all_edu, kind, all_a, all_w, plots_base_dir=plot_dir, epoch=epoch, save=True)

print(f"✅ All plots saved to: {plot_dir}/epoch{epoch}")
