# from main_vanila_model import main


# from multiprocessing import Pool



# a = [1, 2, 3, 4, 5]
# b = [6, 7, 8, 9, 10]

# if __name__ == '__main__':
#     with Pool(18) as pool: # four parallel jobs
#         results = pool.map(main, zip(a, b))




import concurrent.futures
import subprocess
import time

# List of hyperparameter sets


phis = [ 0.0006]
alphas = [1e-1,2e-1, 1e-2, 1.0, 5e-1]

hyperparameter_sets = []
for phi in phis:
    for alpha in alphas:

        # for psi in psis:
            h_set = [
                        '--experiment_title', f'single_H_lr0.1_j1M_100epoch/multiple_r--with_tax--gumbel--phi_{phi}--tau_{alpha}--hard_gumbel',
                        '--alpha_pr', f'{alpha}',
                        '--phi' , f'{phi}',
                        '--hard_gumbel', 
                        '--cuda_no', '0',
                        '--reg_mode', 'two_years',
                        '--num_epochs', '100',
                        '--lmbd', '0',
                        '--save_interval_epoch', '1',
                        '--lr', '1e-1',
                        
                        
                        ]
            hyperparameter_sets.append(h_set)


# print(hyperparameter_sets)
# Path to your training script
script_path = "./main.py"

def run_training(params):
    command = ["python", script_path] + params
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

# Use ThreadPoolExecutor for parallel execution
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit each run_training task to the executor
    for params in hyperparameter_sets:
        # Submit each run_training task to the executor
        future = executor.submit(run_training, params)
        
        # Introduce a delay (e.g., 5 seconds) before starting the next task
        time.sleep(20)
