
import itertools
import subprocess
import pathlib
import concurrent.futures
import torch
import pynvml
import time

# Configuration --------------------

# GPU Setup

USE_ALL_GPUS = True  
PREFERRED_GPU = 1 # 0

NUM_GPUS = torch.cuda.device_count()
required_mem_MB = 8000
MAX_PARALLEL_JOBS_PER_GPU = 4

print(f"Detected {NUM_GPUS} GPU(s). Distributing jobs accordingly.")

def wait_for_available_gpu(gpu_id, required_mem_MB=4000):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    while True:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_MB = mem_info.free / 1024**2
        if free_MB > required_mem_MB:
            break
        print(f"⏳ Waiting for GPU {gpu_id} to free up... ({int(free_MB)} MB available)")
        time.sleep(10)
    pynvml.nvmlShutdown()

# Paths to scripts
train_script = "Train.py"
eval_script = "Eval.py"

# Save
save_dir = "./Experiments/Search_Hyperparameters_LifeTime_wLS_HighLayh"

# Fixed Parameters --------------------

# Economic Enviernment
# psi = 0.01
# phi = 0.00001
# Policy
# SS_Type = "Top_Years"
# SS_Param = 35
SS_Type = "Life_Time"
SS_Param = 0
# DNN 
num_heads = 1
num_hidden_unit_w = 8 #16
num_hidden_unit_r = 3 #5
dropout = 0.0
# Optimization
num_epochs = 150
lr = 1e-3
batch_size = 512
y_gate_temp = 1
# Simulation
num_sim = 20000 #100000
# Randomness
seed_train = 92
seed_eval = 776

# Grid of hyperparameters to search --------------------

# Hyperparemters for serach
# lr = [1e-3, 5e-4, 1e-4]
#batch_size = [1024, 512]
# y_gate_temp = [1, 5, 10]
#lr = [1e-4, 1e-5]
# batch_size = [512, 256]
# y_gate_temp = [1, 5]
#num_hidden_unit_w = [32, 64]
#num_hidden_unit_r = [5, 10]

psi = 0.0001
phi = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10.0,100]

# Create all combinations
#grid_Hyperparameters = list(itertools.product(lr, batch_size, y_gate_temp))
grid_Hyperparameters = list(itertools.product(phi))
#grid_Hyperparameters = list(itertools.product(lr, batch_size, phi))


# Run --------------------



def run_experiment(index, combo):
    #lr, batch_size, y_gate_temp = combo 
    #lr, batch_size, phi = combo 
    phi = combo[0]

    if USE_ALL_GPUS and NUM_GPUS > 0:
        gpu_id = index % NUM_GPUS # Distribute across available GPUs
    else:
        gpu_id = PREFERRED_GPU  # Always use the chosen GPU
    #gpu_id = index % max(1, NUM_GPUS) 

    exp_name = f"lr{lr}_bs{batch_size}_gbt{y_gate_temp}_nh{num_heads}_dw{num_hidden_unit_w}_dr{num_hidden_unit_r}_phi{phi:.0e}_psi{psi}" # Create experiment name
    base_dir = pathlib.Path(save_dir) / exp_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Train command
    train_cmd = [
        "python", train_script,
        "--experiment_title", exp_name,
        "--save_dir", save_dir,
        "--lr", str(lr),
        "--num_sim", str(num_sim), 
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--device", f"cuda:{gpu_id}",
        "--num_hidden_unit_w", str(num_hidden_unit_w),
        "--num_hidden_unit_r", str(num_hidden_unit_r),
        "--SS_Type", SS_Type,
        "--SS_Param", str(SS_Param),
        "--dropout", str(dropout),
        "--psi", str(psi),
        "--phi", str(phi),
        "--seed_train", str(seed_train)
    ]

    # Eval command
    eval_cmd = [
        "python", eval_script,
        "--experiment_title", exp_name,
        "--save_dir", save_dir,
        "--device", f"cuda:{gpu_id}",
        "--num_epochs", str(num_epochs),
        "--num_sim", str(num_sim),
        "--batch_size", str(batch_size),
        #"--num_hidden_unit_w", str(hyperparams["num_hidden_unit_w"]),
        #"--num_hidden_unit_r", str(hyperparams["num_hidden_unit_r"]),
        #"--SS_Type", hyperparams["SS_Type"],
        #"--SS_Param", str(hyperparams["SS_Param"]),
        #"--psi", str(hyperparams["psi"]),
        #"--phi", str(hyperparams["phi"]),
        "--seed_eval", str(seed_eval)
    ]

    try:
        wait_for_available_gpu(gpu_id, required_mem_MB)
        print(f"\n🚀 Starting Experiment [{index+1}/{len(grid_Hyperparameters)}]: {exp_name} on GPU {gpu_id}")
        
        with open(base_dir / "train.log", "w") as fout:
            subprocess.run(train_cmd, stdout=fout, stderr=subprocess.STDOUT, check=True)
        with open(base_dir / "eval.log", "w") as fout:
            subprocess.run(eval_cmd, stdout=fout, stderr=subprocess.STDOUT, check=True)
        with open("grid_search_progress.log", "a") as log:
            log.write(f"[{index+1}/{len(grid_Hyperparameters)}] ✅ Finished: {exp_name}\n")
        with open("global_grid_search.log", "a") as master_log:
            master_log.write(f"[{time.ctime()}] ✅ Finished: {exp_name}\n")
        print(f"✅ Finished: {exp_name}")

    except subprocess.CalledProcessError as e:
        with open("grid_search_progress.log", "a") as log:
            log.write(f"[{index+1}/{len(grid_Hyperparameters)}] ❌ Failed: {exp_name} | Error: {e}\n")

        with open("global_grid_search.log", "a") as master_log:
            master_log.write(f"[{time.ctime()}] ❌ Failed: {exp_name} | Error: {e}\n")

        print(f"❌ Failed: {exp_name} | Error: {e}")


# Set parallel jobs to match number of GPUs or cap it if needed

if USE_ALL_GPUS and NUM_GPUS > 0:
    MAX_PARALLEL_JOBS = NUM_GPUS * MAX_PARALLEL_JOBS_PER_GPU
else:
    MAX_PARALLEL_JOBS = MAX_PARALLEL_JOBS_PER_GPU 

# Run jobs in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
    futures = [executor.submit(run_experiment, idx, combo) for idx, combo in enumerate(grid_Hyperparameters)]
    for future in concurrent.futures.as_completed(futures):
        future.result()
