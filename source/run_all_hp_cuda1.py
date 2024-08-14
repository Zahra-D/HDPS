import concurrent.futures  
import subprocess  
from itertools import product  
    # parser.add_argument('--start_lr_decay', type=int, default=50)#[50, 70]
    # parser.add_argument('--lr_sch_gamma_decay', type=float, default=.9) #[.9, .8]
    # parser.add_argument('--lr_scheduler', type=str, default='exp')# [exp, step]
    # parser.add_argument('--lr_sch_step_size', type=int, default=5)#do not change
    # parser.add_argument('--lr_sch_step_decay', type = float, default=.5)# [0.5, .2]

    # parser.add_argument('--r_pr', type=float, default=1e-3) #[1e-3, 2e-3, 5e-3]
    # parser.add_argument('--r_h', type=float, default=1e-3) #[1e-3, 2e-3, 5e-3]
    # parser.add_argument('--per_step_update_h', type=int, default=10) #[10,50,100]
    # parser.add_argument('--per_step_update_pr', type=int, default=10) #[10,50,100]
hyperparameter_options = {  
    
    'r_pr': [2e-3], 
    'r_h': [1e-3, 2e-3, 5e-3],
    'per_step_update_pr': [10,100, 500], 
    'per_step_update_h': [10,100, 500],  
    'start_lr_decay': [20, 70],    
}  




scheduler_step_params = hyperparameter_options.copy() 
scheduler_step_params['lr_sch_step_decay']  =  [0.5, 0.2]
scheduler_step_params['lr_scheduler'] = ['step']

scheduler_exp_params = hyperparameter_options.copy() 
scheduler_exp_params['lr_sch_gamma_decay'] = [.9, .8]
scheduler_exp_params['lr_scheduler'] = ['exp']

lr_scheduler_type = ['exp', 'step']  # Types of schedulers  




hyperparameter_combinations = []  
for s_type in lr_scheduler_type:  
    if s_type == 'exp':  
        hyperparameter_combinations.extend(list(product(*scheduler_exp_params.values())))
    else:  # "scheduler_b"  
        hyperparameter_combinations.extend(list(product(*scheduler_step_params.values())))

# Generate all combinations of hyperparameters  
# hyperparameter_combinations = list(product(*hyperparameter_options.values()))  
len(hyperparameter_combinations)
# Path to your training script  
script_path = "./main.py"  

def run_training(params):  
    command = ["python", script_path] + params  
    try:  
        subprocess.run(command, check=True)  
    except subprocess.CalledProcessError as e:  
        print(f"Error running command with params {params}: {e}")  

# Convert tuples of parameters to the command-line arguments format  
def format_params(params):  
    return [  
        '--experiment_title', f"HP_searching/{params[-1]}_scheduler/r_pr{params[0]}--r_h{params[1]}--N_r{params[2]}--N_h{params[3]}--start_lr_decay{params[4]}--lr_sch_step_decay{params[5]}--",
        '--r_pr', f'{params[0]}',  
        '--r_h', f'{params[1]}',  
        '--per_step_update_pr', f'{params[2]}',  
        '--per_step_update_h', f'{params[3]}',  
        '--start_lr_decay', f'{params[4]}', 
        '--lr_sch_step_decay', f'{params[5]}', 
        '--lr_scheduler', f'{params[6]}', 
        
        
        '--num_epochs', '150',
        '--cuda_no', '1',  
        '--reg_mode', 'two_years'  
    ]  

# Use ThreadPoolExecutor for parallel execution  
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  
    futures = {}  
    
    for params in hyperparameter_combinations:  
        # Format the parameters for the command line  
        formatted_params = format_params(params)  
        
        # Submit the training task to the executor  
        future = executor.submit(run_training, formatted_params)  
        futures[future] = params  # Keep track of which params were used  
        
        # Optional: Print status for every 4 submitted tasks  
        if len(futures) % 4 == 0:  
            print(f"Submitted {len(futures)} tasks...")  

    # Wait for all the futures to complete  
    for future in concurrent.futures.as_completed(futures):  
        params = futures[future]  
        try:  
            future.result()  # Raise an exception if the run failed  
        except Exception as e:  
            print(f"Run failed for params: {params}, error: {e}")  

print("All training runs completed.")