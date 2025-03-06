import torch
import torch.optim as optim
import numpy as np
import random
import pathlib
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from imports import *
from Parameters import *
from functions import *
from utils import *
from model import Model

# **Set Hyperparameters Directly**
hyperparams = {
    "batch_size": 10000,
    "seed": 92,
    "num_hidden_unit_w": 10,
    "num_hidden_unit_r": 5,
    "num_epochs": 1000,
    "reg_mode": "each10",
    "lr": 1e-3,
    "lmbd": 1e-2,
    "psi": 0.04,
    "phi": 0.0006,
    "alpha_pr": 5,
    "save_interval_epoch": 100,
    "save_dir": "./Experiments"
}

# **Set random seeds for reproducibility**
torch.manual_seed(hyperparams["seed"])
np.random.seed(hyperparams["seed"])
random.seed(hyperparams["seed"])

# **Select Device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Generate Training Dataset**
dataset_train = generating_dataset(J, T_LR - AGE_0, THETA_0, P_EDU)
dataloader_train = DataLoader(dataset_train, batch_size=hyperparams["batch_size"], shuffle=True)

# **Initialize Model and Optimizer**
model = Model(num_hidden_node_w=hyperparams["num_hidden_unit_w"], alpha_pr=hyperparams["alpha_pr"]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"])

# **Create Directories for Saving Results**
base_dir = pathlib.Path(f'{hyperparams["save_dir"]}/TrainingResults')
base_dir.mkdir(parents=True, exist_ok=True)
(saved_model_dir := base_dir / "model").mkdir(parents=True, exist_ok=True)
(saved_run_dir := base_dir / "runs").mkdir(parents=True, exist_ok=True)

# **Save Hyperparameters**
with open(base_dir / "hyperparams.json", "w") as f:
    json.dump(hyperparams, f, indent=4)

# **Initialize Logger**
writer = SummaryWriter(saved_run_dir)

# **Training Function**
def train_step(model, dataloader, epoch, writer, optimizer):
    model.train()
    train_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{hyperparams['num_epochs']}", unit="batch", leave=False)
    
    for batch in train_iterator:
        theta_t_, w_t_, edu_ = batch
        w_t_ = w_t_.to(device)
        len_batch = len(batch[0])
        a_0 = torch.tensor([A_0] * len_batch, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        all_a, all_c, all_c_ER, all_pr_bar, all_pr, all_h, all_y = model(theta_t_.to(device), edu_.to(device), a_0, w_t_)

        loss = loss_function_retirement_pr_cross(model, all_c, all_c_ER, all_pr_bar, all_pr, all_h, epoch, writer, hyperparams)
        writer.add_scalar("Loss/all", loss.item(), epoch)

        loss.backward()
        optimizer.step()
        loss.detach()
        train_iterator.set_postfix(loss=loss.item())

# **Training Loop**
for epoch in range(hyperparams["num_epochs"]):
    train_step(model, dataloader_train, epoch, writer, optimizer)
    if epoch % hyperparams["save_interval_epoch"] == 0:
        save_checkpoint(model, optimizer, saved_model_dir, epoch)

# **Save Final Model**
save_checkpoint(model, optimizer, saved_model_dir, hyperparams["num_epochs"] - 1)

print("Training complete. Results saved in:", base_dir)






# # SQ Model  --------------------


# # Setting Random Seeds
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)
    
# arg_dict = vars(args) # Converting args to a Dictionary
    
# # Generating Train Dataset
# dataset_train = generating_dataset(J,  T_LR-AGE_0, THETA_0, P_EDU) # create exogenous states for training data.
# dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True) # Wraps it in DataLoader for batch processing
    
# # Selecting the Computing Device
# if args.device == 'cuda':
#     device = torch.device(f"cuda:{args.cuda_no}")
# elif args.device == 'cpu':
#     device = 'cpu'
# num_epochs = args.num_epochs

# # Defining the Optimizer
# optimizer_func = optim.AdamW # Uses AdamW optimizer 
    
# # Learning Rate Hyperparameter Tuning
    
# if (args.lr == 0):     # 0 means we want to check all the predefined learning rates
#     lrs = [1e-1, 1e-2, 1e-3]
# else:
#     lrs = [args.lr]
# for lr in lrs: # Looping Over Hyperparameter lr 

#     # Number of Hidden Units Hyperparameter Tuning
        
#     if (args.num_hidden_unit_w == 0):         # 0  means we want to check all the predefined nhu.
#         num_h_u_w = [30,10]
#     else:
#         num_h_u_w = [args.num_hidden_unit_w]  
#     for num_h_u in num_h_u_w: # Looping Over Hyperparameter nh_w

#         #  Initializing the Model
#         model = Model(num_hidden_node_w=num_h_u, alpha_pr= args.alpha_pr)
#         optimizer = optimizer_func(model.parameters(), lr=lr)
#         model.to(device)

            
#         # Saving Results of each Experiment
            
#         # Setting Up Directories
#         base_dir  = pathlib.Path(
#             f'{args.save_dir}/{args.experiment_title}'\
#             f'/base_model_with_regu_wb_{args.reg_mode}_{num_h_u}HiddenUnits_seed{args.seed}_phi{args.phi}'\
#             f'/{args.batch_size}_batch_size'\
#             f'/PSI{args.psi}'\
#             f'/lambda{args.lmbd}'\
#             f'/{optimizer_func.__name__}_lr:{lr}'
#         )

#         # Creating Necessary Directories
#         saved_model_dir = base_dir / "model" # Folder for Model Checkpoints 
#         saved_plot_dir =  base_dir / "plot" # Folder for Graphs
#         saved_run_dir = base_dir / "runs" # flder for Logging Directory
            
#         # Ensures folders exist before training.
#         base_dir.mkdir(parents=True, exist_ok=True) 
#         saved_model_dir.mkdir(parents=True, exist_ok=True)
#         saved_plot_dir.mkdir(parents=True, exist_ok=True)

#         # Save experiment settings as a JSON file.
#         args_dict = vars(args)
#         with open(f'{base_dir}/arguments.json', 'w') as file: 
#             json.dump(args_dict, file, indent=4)

#         # Setting Up a TensorBoard Logger
#         writer = SummaryWriter(saved_run_dir)
            
#         # Training Loop
            
#         for epoch in range(num_epochs):
#             train_step(model, dataloader_train, epoch, writer, optimizer, device, args)
#             # Saving Model Checkpoints
#             if (epoch%args.save_interval_epoch)==0: # at every save_interval_epoch epochs.
#                 save_checkpoint(model, optimizer, saved_model_dir, epoch)
#         # Saving Final Model        
#         save_checkpoint(model, optimizer, saved_model_dir, epoch)
