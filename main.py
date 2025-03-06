# Run the training and evaluation --------------------

from imports import *
from Parameters import *
from functions import *
from utils import *
from model import Model


# Traning the model --------------------


def train_step(model, dataloader, epoch, s_writer, optimizer, device, args):

    # Set model to training mode
    model.train()
    # Initialize progress bar
    train_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch", leave=False)
    # Loop through the dataloader batch by batch.
    for batch_idx, batch in enumerate(train_iterator):
        
        # Exogenous inputs
        theta_t_, w_t_, edu_ = batch # Extract batch features
        w_t_ = w_t_.to(device)
        len_batch = len(batch[0]) # Determine batch size
        a_0 = torch.tensor([A_0]* len_batch, dtype=torch.float32) # Initial asset for all batch

        # Reset gradients
        optimizer.zero_grad()  # Clears previous gradients from the optimizer

        # Forward Pass
        all_a, all_c, all_c_ER, all_pr_bar, all_pr, all_h, all_y = model(theta_t_.to(device), edu_.to(device), a_0.to(device),w_t_)

        # Computing the Loss
        loss = loss_function_retirement_pr_cross(model, all_c, all_c_ER, all_pr_bar, all_pr, all_h, epoch, s_writer, args)
        
        # Logging the Loss
        s_writer.add_scalar('Loss/all', loss.item(), epoch)

        # torch.autograd.set_detect_anomaly(True)
        loss.backward() # # Backpropagation 
        optimizer.step() # Updating Model Parameters
        loss.detach().cpu() # Detaching Loss from Computational Graph
        train_iterator.set_postfix(loss=loss.item()) # Updating the Progress Bar


# Evaluation and Saving  --------------------

# evaluates the trained model on the dataset without updating its parameters. 
# It also saves results for further analysis and generates plots for visualization.

def do_eval_save(model, dataloader, base_dir, epoch, device, s_writer,args):

    # Initializing Lists
    all_w = []
    all_a = []
    all_h = []
    all_y = []
    all_c = []
    all_t = []
    all_edu = []
    all_theta = []

    # Set Model to Evaluation Mode
    model.eval() # Disables dropout and batch normalization updates.

    # Iterating Over the Evaluation Dataset
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad(): # disable gradient computation

            # Extracting exogenous inputs
            theta_t, w_t, edu = batch
            w_t = w_t.to(device)
            len_batch = len(batch[0])
            a_0 = torch.tensor([A_0]* len_batch)

            # Running Model Inference
            retirement_p, working_c, retirement_c, working_a, retirement_a, working_h, retirement_h, all_y_, all_r  = model(theta_t.to(device), edu.to(device), a_0.to(device), w_t)

            # Combining Working and Retirement States
            L = retirement_c.shape[1]
            r_t = (torch.concat([all_r>.5, torch.ones(len_batch, 1).to(device)], dim=-1)).argmax(dim=-1)
            a_t = torch.concat([working_a, retirement_a.transpose(1,2)[torch.arange(len_batch), r_t]], dim =1)
            c_t = torch.concat([working_c, retirement_c.transpose(1,2)[torch.arange(len_batch), r_t]], dim =1)
            h_t = torch.concat([working_h, retirement_h.transpose(1,2)[torch.arange(len_batch), r_t]], dim =1)

            # Storing Results
            all_theta.extend(theta_t.cpu())
            all_w.extend(w_t.cpu())
            all_edu.extend(edu.cpu())
            all_a.extend(a_t.cpu())
            all_h.extend(h_t.cpu())
            all_y.extend(all_y_.cpu())
            all_c.extend(c_t.cpu())
            all_t.extend(r_t.cpu())
            
    # Stacking Results for Analysis
    all_w = torch.stack(all_w) # Combines lists into tensors for easier manipulation.
    all_a = torch.stack(all_a)
    all_h = torch.stack(all_h)
    all_y = torch.stack(all_y)  
    all_c = torch.stack(all_c)
    all_theta = torch.stack(all_theta)  

    # Generating Plots
    #draw_all_plots(base_dir, all_a, all_h, all_w, all_theta, all_c, all_y, epoch)


# Running Experiments  --------------------


def main(args):

    # Setting Random Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    arg_dict = vars(args) # Converting args to a Dictionary
    
    # Generating Train Dataset
    dataset_train = generating_dataset(J,  T_LR-AGE_0, THETA_0, P_EDU) # create exogenous states for training data.
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True) # Wraps it in DataLoader for batch processing
    
    # Selecting the Computing Device
    if args.device == 'cuda':
        device = torch.device(f"cuda:{args.cuda_no}")
    elif args.device == 'cpu':
        device = 'cpu'
    num_epochs = args.num_epochs

    # Defining the Optimizer
    optimizer_func = optim.AdamW # Uses AdamW optimizer 
    
    # Learning Rate Hyperparameter Tuning
    
    if (args.lr == 0):     # 0 means we want to check all the predefined learning rates
        lrs = [1e-1, 1e-2, 1e-3]
    else:
        lrs = [args.lr]
    for lr in lrs: # Looping Over Hyperparameter lr 

        # Number of Hidden Units Hyperparameter Tuning
        
        if (args.num_hidden_unit_w == 0):         # 0  means we want to check all the predefined nhu.
            num_h_u_w = [30,10]
        else:
            num_h_u_w = [args.num_hidden_unit_w]  
        for num_h_u in num_h_u_w: # Looping Over Hyperparameter nh_w

            #  Initializing the Model
            model = Model(num_hidden_node_w=num_h_u, alpha_pr= args.alpha_pr)
            optimizer = optimizer_func(model.parameters(), lr=lr)
            model.to(device)

            
            # Saving Results of each Experiment
            
            # Setting Up Directories
            base_dir  = pathlib.Path(
                f'{args.save_dir}/{args.experiment_title}'\
                f'/base_model_with_regu_wb_{args.reg_mode}_{num_h_u}HiddenUnits_seed{args.seed}_phi{args.phi}'\
                f'/{args.batch_size}_batch_size'\
                f'/PSI{args.psi}'\
                f'/lambda{args.lmbd}'\
                f'/{optimizer_func.__name__}_lr:{lr}'
            )

            # Creating Necessary Directories
            saved_model_dir = base_dir / "model" # Folder for Model Checkpoints 
            saved_plot_dir =  base_dir / "plot" # Folder for Graphs
            saved_run_dir = base_dir / "runs" # flder for Logging Directory
            
            # Ensures folders exist before training.
            base_dir.mkdir(parents=True, exist_ok=True) 
            saved_model_dir.mkdir(parents=True, exist_ok=True)
            saved_plot_dir.mkdir(parents=True, exist_ok=True)

            # Save experiment settings as a JSON file.
            args_dict = vars(args)
            with open(f'{base_dir}/arguments.json', 'w') as file: 
                json.dump(args_dict, file, indent=4)

            # Setting Up a TensorBoard Logger
            writer = SummaryWriter(saved_run_dir)
            
            # Training Loop
            
            for epoch in range(num_epochs):
                train_step(model, dataloader_train, epoch, writer, optimizer, device, args)
                # Saving Model Checkpoints
                if (epoch%args.save_interval_epoch)==0: # at every save_interval_epoch epochs.
                    save_checkpoint(model, optimizer, saved_model_dir, epoch)
            # Saving Final Model        
            save_checkpoint(model, optimizer, saved_model_dir, epoch)


# Command-Line Arguments  --------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    
    # Step 3: Add arguments
    # Saved_stuff/multipleBlockR--pr--weighted_sum_H/
    parser.add_argument('--experiment_title', type=str)
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=92,  help='random seed for reproducibility')
    parser.add_argument('--num_hidden_unit_w', default=10, type=int, help='number of hidden units for the working network')
    parser.add_argument('--num_hidden_unit_r', default=5, type=int, help='number of hidden units for the retirement network')
    parser.add_argument('--num_epochs', type=int, default=1000,  help='number of epochs')
    parser.add_argument('--reg_mode', type=str, choices=['each10', 'last_year'],default='each10', help='Choose a regularization mode (each10,last_year)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lmbd', type=float, default=1e-2, help='retirement lambda')
    parser.add_argument('--psi', type=float, default=0.04, help='work hour disutility coefficient')
    parser.add_argument('--save_interval_epoch', type=int, default=100, help='Number of epochs between saving model checkpoints during training')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_no', type=int, choices=[0,1],default=0, help='the index of cuda device to run the code on it')
    parser.add_argument('--save_dir', type=str,default='./Experiments', help='the directory that the training result will be saved in')
    parser.add_argument('--phi', type=float, default=0.0006, help='Phi in utility')
    parser.add_argument('--alpha_pr', type=float, default=5, help='the slop of sigmoid funciotn of pr')
 
    # Parsing the Arguments
    args = parser.parse_args()
    
    # Running the main Function
    main(args)
