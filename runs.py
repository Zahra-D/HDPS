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


phis = [1, 0,  .1, .001, ]
alphas = [1,10, 1000]
hyperparameter_sets = []
for phi in phis:
    for alpha in alphas:
        # for psi in psis:
            h_set = [
                        '--experiment_title', f'single_r--alpha_pr_{alpha}--phi_{phi}',
                        '--alpha_pr', f'{alpha}'   ,
                        '--phi' , f'{phi}',
                        '--cuda_no', '1'
                        
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
        time.sleep(30)
