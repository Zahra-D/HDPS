
from imports import *
from Parameters import *
from Functions import *
from Model import *

import argparse
import pathlib
import json
import random
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.serialization
from tqdm import tqdm

# Set Hyperparameters --------------------

from Model import Master_Model, Working_Model, Retiree_Model

parser = argparse.ArgumentParser()

# Experiment
parser.add_argument("--experiment_title", type=str, default="default_experiment")
# Optimization Hyperparemters
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_sim", type=int, default=1000000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed_train", type=int, default=92)
# DNN Architecture
parser.add_argument("--num_hidden_unit_w", type=int, default=16)
parser.add_argument("--num_hidden_unit_r", type=int, default=5)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--y_gate_temp", type=int, default=10)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--SS_Type", type=str, default="Top_Years")  # or 'Life_Time', 'Non_Parametric'
parser.add_argument("--SS_Param", type=int, default=35)
# Transfer/Warming/Freezing Learning
parser.add_argument("--transfer_learning_type", type=str, choices=["none", "plain", "bilevel", "warm", "freeze_res", "bilevel_freeze_res", "staged_core_then_res","ret_first","Mult_Lr", "ret_first_freeze_res"], default="none")
parser.add_argument("--pretrained_path", type=str, default="./Experiments/Numerical_Exercise/model/epoch250/model.pt")
parser.add_argument("--freeze_epochs", type=int, default=0)      # Only for bilevel
parser.add_argument("--warm_epochs", type=int, default=0)        # Only for warm
parser.add_argument("--warm_coef", type=float, default=0.01)     # Only for warm
parser.add_argument("--freeze_res_epochs", type=int, default=0)
parser.add_argument("--freeze_res_epochs_c", type=int, default=0)  # When to unfreeze consumption residuals
parser.add_argument("--freeze_res_epochs_h", type=int, default=0)  # When to unfreeze hours residuals
parser.add_argument("--warmup_epochs_lr", type=int, default=10)    # Learning Rate Warmup # How many epochs to warm up
parser.add_argument("--warmup_init_lr", type=float, default=1e-5)  # Learning Rate Warmup # Start LR for newly unfrozen blocks
parser.add_argument("--freeze_epochs_ret", type=int, default=20)   # Used only for ret_first
parser.add_argument("--lr_retiree", type=float, default=2e-4)      # for Mult_Lr
parser.add_argument("--lr_working", type=float, default=5e-5)      # for Mult_Lr
# Gumbel-Softmax Temperature Annealing
parser.add_argument("--gumbel_temp_min", type=float, default=0.5)     # τ_min
parser.add_argument("--gumbel_temp_decay", type=float, default=1e-5)  # r
parser.add_argument("--gumbel_temp_update_every", type=int, default=500)  # N
parser.add_argument("--soft_rank_temp", type=float, default=0.1, help="Temperature for soft ranking (Top_Years)")


# Economic Envienment
parser.add_argument("--psi", type=float, default=0.01)
parser.add_argument("--phi", type=float, default=0.0006)
# Report & Save
parser.add_argument("--save_interval_epoch", type=int, default=50)
parser.add_argument("--save_dir", type=str, default="./Experiments")

args = parser.parse_args()


## Configuration --------------------


# Set random seeds
torch.manual_seed(args.seed_train)
np.random.seed(args.seed_train)
random.seed(args.seed_train)

# Select Device
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Save directory
save_dir = pathlib.Path(args.save_dir)
base_dir = save_dir / args.experiment_title

model_dir = base_dir / "model"
run_dir = base_dir / "runs"
base_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
run_dir.mkdir(parents=True, exist_ok=True)

with open(base_dir / "hyperparams.json", "w") as f:
    json.dump(vars(args), f, indent=4)

writer = SummaryWriter(run_dir)


## Data --------------------


""" Generate Training Dataset """

dataset_train = generating_dataset(args.num_sim, T_W, THETA_0, P_EDU)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)


## Model --------------------

""" Initialize Model and Optimizer """

working_model = Working_Model(
    SS_Type=args.SS_Type,
    SS_Param=args.SS_Param,
    d_model=args.num_hidden_unit_w,
    num_heads=args.num_heads,
    dropout=args.dropout,
    y_gate_temp=args.y_gate_temp,
    soft_rank_temp=args.soft_rank_temp,
).to(device)

retiree_model = Retiree_Model(num_hidden_units_R=args.num_hidden_unit_r).to(device)

model = Master_Model(
    working_model=working_model,
    retiree_model=retiree_model,
).to(device)

# transfer learning

if args.transfer_learning_type != "none":
    if pathlib.Path(args.pretrained_path).exists():
        print(f"🔁 Loading pretrained model from {args.pretrained_path}")
        with torch.serialization.safe_globals({
            'Master_Model': Master_Model,
            'Working_Model': Working_Model,
            'Retiree_Model': Retiree_Model
        }):
            pre_model = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        # Ensure model classes are already imported (from Model.py)
        #pre_model = torch.load(args.pretrained_path, map_location=device)
        pre_state = pre_model.state_dict()
        #loaded = model.load_state_dict(pre_state, strict=False)
        model_state = model.state_dict()
        ok_state = {k: v for k, v in pre_state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model_state.update(ok_state)
        model.load_state_dict(model_state)  

        if args.transfer_learning_type == "plain":
            print("✅ Plain transfer learning: using pretrained weights for init")
        
        elif args.transfer_learning_type == "bilevel":
            # Freeze all non-h-task parameters
            for name, param in model.named_parameters():
                if "task_layer_h_" not in name:
                    param.requires_grad = False
            print("🧊 Bi-level learning: frozen non-h params")

        elif args.transfer_learning_type == "warm":
            print(f"🔥 Warm-start: using pretrained init, h penalty for {args.warm_epochs} epochs")
        elif args.transfer_learning_type == "freeze_res":
            # 1) zero-initialise every residual head parameter
            for name, param in model.named_parameters():
                if "task_layer_h_residual" in name or "task_layer_x_residual" in name:
                    nn.init.zeros_(param.data)
                    param.requires_grad = False          # 2) lock them
            print(f"🚫  Residual heads frozen for {args.freeze_res_epochs} epochs")

        elif args.transfer_learning_type == "bilevel_freeze_res":
            for name, param in model.named_parameters():
                if "task_layer_x_" in name:
                    param.requires_grad = True  # x_core always trains
                elif "task_layer_h_" in name or "task_layer_x_residual" in name:
                    param.requires_grad = False
            print(f"🧊 Bilevel+FreezeRes: x_core trains first, rest frozen")

        elif args.transfer_learning_type == "staged_core_then_res":
            for name, param in model.named_parameters():
                if "task_layer_x_core" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("🧊 Staged Training: Start with x_core only")

        elif args.transfer_learning_type == "ret_first":
            # We freeze working_model for the first few epochs later in train_step()
            print(f"🧊 Ret_First: working_model will be frozen for {args.freeze_epochs_ret} epochs")

        elif args.transfer_learning_type == "ret_first_freeze_res":
            print(f"🧊 Ret_First+FreezeRes: Working model frozen for {args.freeze_epochs_ret} epochs; x_core starts first, x_res unfrozen after {args.freeze_res_epochs_c} epochs")


    else:
        print(f"❌ Pretrained model not found at {args.pretrained_path}")


## Train the Model --------------------

model.training_args = args
model.working_model.training_args = args  # So Working_Model has access to training args
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()), 
#     lr=args.lr
# )
param_groups = []
# x_core (trains from beginning)
x_core_params = [p for n, p in model.named_parameters() if "task_layer_x_core" in n]
param_groups.append({"params": x_core_params, "lr": args.lr})
# h_core (unfrozen later)
h_core_params = [p for n, p in model.named_parameters() if "task_layer_h_core" in n]
param_groups.append({"params": h_core_params, "lr": args.warmup_init_lr, "name": "h_core", "unfreeze_epoch": args.freeze_epochs})
# x_res
x_res_params = [p for n, p in model.named_parameters() if "task_layer_x_residual" in n]
param_groups.append({"params": x_res_params, "lr": args.warmup_init_lr, "name": "x_res", "unfreeze_epoch": args.freeze_res_epochs_c})
# h_res
h_res_params = [p for n, p in model.named_parameters() if "task_layer_h_residual" in n]
param_groups.append({"params": h_res_params, "lr": args.warmup_init_lr, "name": "h_res", "unfreeze_epoch": args.freeze_epochs + args.freeze_res_epochs_h})
# Retiree model (train independently)
retiree_params = [p for n, p in model.named_parameters() if "retiree_model" in n]
param_groups.append({"params": retiree_params, "lr": args.lr_retiree})

if args.transfer_learning_type == "Mult_Lr":
    working_params = [p for n, p in model.named_parameters() if "working_model" in n]
    retiree_params = [p for n, p in model.named_parameters() if "retiree_model" in n]

    param_groups = [
        {"params": working_params, "lr": args.lr_working},
        {"params": retiree_params, "lr": args.lr_retiree},
    ]

    print(f"💡 Using Mult_Lr: working_model lr={args.lr_working}, retiree_model lr={args.lr_retiree}")



optimizer = torch.optim.AdamW(param_groups)

# Multi learning rate
def apply_blockwise_lr(epoch, block_name, base_lr, unfreeze_epoch):
    warmup_epochs = args.warmup_epochs_lr
    if epoch < unfreeze_epoch:
        return args.warmup_init_lr
    elif epoch >= unfreeze_epoch + warmup_epochs:
        return base_lr
    else:
        rel_epoch = epoch - unfreeze_epoch
        return args.warmup_init_lr + (base_lr - args.warmup_init_lr) * (rel_epoch / warmup_epochs)

# dynamic Gumbel-Softmax temperature annealing (Jang et al. (2017))
def get_dynamic_temperature(global_step, r, tau_min, update_every):
    if global_step % update_every != 0:
        return None  # Don't update τ if not the right step
    return max(tau_min, math.exp(-r * global_step))


# Training 
def train_step(model, dataloader, epoch, writer, optimizer):
    
    global_step = epoch * len(dataloader)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)

    # Dynamically adjust LR during warmup (before batch loop!)
    for group in optimizer.param_groups:
        if "name" not in group:
            continue
        block = group["name"]
        group["lr"] = apply_blockwise_lr(epoch, block, args.lr, group["unfreeze_epoch"])

        # log learning rate once per epoch (instead of per batch)
        writer.add_scalar(f"LR/{block}", group["lr"], epoch)

    for batch in iterator:

        theta, _, edu = batch
        B = len(theta)
        a0 = torch.full((B,), A_0).to(device)

        # Update temperture parameter
        curr_tau = get_dynamic_temperature(
            global_step,
            r=args.gumbel_temp_decay,
            tau_min=args.gumbel_temp_min,
            update_every=args.gumbel_temp_update_every,
        )
        if curr_tau is None:
            curr_tau = model.working_model.y_gate_temp  # fallback to static value

        global_step += 1

        writer.add_scalar("Train/Gumbel_Temperature", curr_tau, global_step)

        # 🧊 Unfreeze at the correct epoch
        if args.transfer_learning_type == "bilevel" and epoch == args.freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            print(f"🟢 Unfrozen all params at epoch {epoch}")

        if args.transfer_learning_type == "freeze_res" and epoch == args.freeze_res_epochs:
            for name, p in model.named_parameters():
                if "task_layer_h_residual" in name or "task_layer_x_residual" in name:
                    p.requires_grad = True
            print(f"🟢 Residual heads unfrozen at epoch {epoch}")

        if args.transfer_learning_type == "bilevel_freeze_res":
            if epoch == args.freeze_epochs:
                for name, p in model.named_parameters():
                    if "task_layer_h_" in name and "residual" not in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze h_core at epoch {epoch}")
            if epoch == args.freeze_res_epochs_c:
                for name, p in model.named_parameters():
                    if "task_layer_x_residual" in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze x_res at epoch {epoch}")
            if epoch == args.freeze_epochs + args.freeze_res_epochs_h:
                for name, p in model.named_parameters():
                    if "task_layer_h_residual" in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze h_res at epoch {epoch}")

        if args.transfer_learning_type == "staged_core_then_res":
            if epoch == args.freeze_epochs:
                for name, p in model.named_parameters():
                    if "task_layer_h_core" in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze h_core at epoch {epoch}")
            if epoch == args.freeze_epochs + args.freeze_res_epochs_c:
                for name, p in model.named_parameters():
                    if "task_layer_x_residual" in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze x_res at epoch {epoch}")
            if epoch == args.freeze_epochs + args.freeze_res_epochs_c + args.freeze_res_epochs_h:
                for name, p in model.named_parameters():
                    if "task_layer_h_residual" in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze h_res at epoch {epoch}")

        if args.transfer_learning_type == "ret_first_freeze_res":
            if epoch == args.freeze_res_epochs_c + args.freeze_epochs_ret:
                for name, p in model.named_parameters():
                    if "task_layer_x_residual" in name:
                        p.requires_grad = True
                print(f"🟢 Unfroze x_res at epoch {epoch}")


        optimizer.zero_grad()

        out = model.simulate(theta.to(device), edu.to(device), a0, tau=curr_tau)

        c_t = out["c_sim_t"]
        h_t = out["h_sim_t"]

        c_mean = c_t.mean().item()
        h_mean = h_t.mean().item()

        writer.add_scalar("Train/Mean_Consumption", c_mean, epoch)
        writer.add_scalar("Train/Mean_Hours", h_mean, epoch)

        pre_retirement_c = c_t[:, :T_W]
        post_retirement_c = c_t[:, T_W:]
        pre_retirement_h = h_t[:, :T_W]
        post_retirement_h = h_t[:, T_W:]

        writer.add_scalar("Train/Mean_Consumption_BeforeRet", pre_retirement_c.mean().item(), epoch)
        writer.add_scalar("Train/Mean_Consumption_AfterRet", post_retirement_c.mean().item(), epoch)
        writer.add_scalar("Train/Mean_Hours_BeforeRet", pre_retirement_h.mean().item(), epoch)
        writer.add_scalar("Train/Mean_Hours_AfterRet", post_retirement_h.mean().item(), epoch)

        c_age_means = c_t.mean(dim=0, keepdim=True)  # shape [1, T]
        spike_mask = c_t > (3.0 * c_age_means)       # shape [B, T], bool
        spike_fraction = spike_mask.sum().item() / c_t.numel() # Total number of spike events across all people and times

        writer.add_scalar("Train/Fraction_Consumption_Spikes", spike_fraction, epoch)

        c_age_mean = c_t.mean(dim=0)  # shape [T]
        c_overall_mean = c_t.mean()
        collapse_years = (c_age_mean < 0.25 * c_overall_mean).float() # Flag years where age-mean is too low
        collapsed_year_fraction = collapse_years.mean().item() # Compute fraction of years that are collapsed

        writer.add_scalar("Train/Fraction_Collapsed_Years", collapsed_year_fraction, epoch)
        
        c_jump = (c_age_mean[1:] - c_age_mean[:-1]).abs().max().item()
        
        writer.add_scalar("Train/Max_Interyear_Consumption_Jump", c_jump, epoch)

        args_simple = argparse.Namespace(phi=args.phi, psi=args.psi)

        loss = loss_function_fixed_retirement(c_t, h_t, epoch, writer, args_simple)
        
        # Warm-start penalty
        if args.transfer_learning_type == "warm" and epoch < args.warm_epochs:
            h_penalty = (h_t.mean() - H_FT) ** 2
            loss += args.warm_coef * h_penalty


        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/all", loss.item(), epoch)
        writer.flush()
        iterator.set_postfix(loss=loss.item(), c_mean = c_mean, h_mean = h_mean)



# Main Loop 
for epoch in range(args.num_epochs):

    if args.transfer_learning_type in ["ret_first", "ret_first_Mult_Lr", "ret_first_freeze_res"]:
        if epoch < args.freeze_epochs_ret:
            for param in model.working_model.parameters():
                param.requires_grad = False
            if epoch == 0:
                print(f"🧊 Working model frozen for first {args.freeze_epochs_ret} epochs")
        elif epoch == args.freeze_epochs_ret:
            for param in model.working_model.parameters():
                param.requires_grad = True
            print(f"🟢 Working model unfrozen at epoch {epoch}")

    train_step(model, dataloader_train, epoch, writer, optimizer)
    if epoch % args.save_interval_epoch == 0:
        save_checkpoint(model, optimizer, model_dir, epoch)

# Save final
save_checkpoint(model, optimizer, model_dir, args.num_epochs)
print(f"✅ Training complete. Results saved at {base_dir}")

