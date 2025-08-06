
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
PREFERRED_GPU = 0 # 0

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
save_dir = "./Experiments/Search_Hyperparameters_35TopYears_DyTemp"

# Fixed Parameters --------------------

fixed = {

    # Economic Enviernment
    "psi": 0.001,
    "phi": 0.001,
    # Policy
    "SS_Type": "Top_Years",
    "SS_Param": 35,
    # DNN
    "num_heads": 2,
    "num_hidden_unit_w": 32,
    "num_hidden_unit_r": 3,
    "dropout": 0.0,
    # Optimization 
    "num_epochs": 100,
    "lr": 5e-5,
    "batch_size": 512,
    # Learning
    "transfer_learning_type": "ret_first_freeze_res",
    "pretrained_path": "./Experiments/Numerical_Exercise_LifeTimes_NoLS_LowHidLay_HighwayR_MidEpoch_Highlr/model/epoch100/model.pt",
    "warm_epochs": 50,
    "warm_coef": 0.01,
    "warmup_epochs_lr": 50,           
    "warmup_init_lr": 1e-6,      
    "freeze_res_epochs": 25,
    "freeze_epochs": 25,
    "freeze_res_epochs_c": 20,
    "freeze_res_epochs_h": 10,
    "freeze_epochs_ret": 5,
    "lr_working": 5e-5,
    "lr_retiree": 1e-4,
    # Simulation
    "num_sim": 25000,
    # Randomness
    "seed_train": 92,
}

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

#psi = 0.0001
#phi = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10.0,100]

# Create all combinations
#grid_Hyperparameters = list(itertools.product(lr, batch_size, y_gate_temp))
#grid_Hyperparameters = list(itertools.product(phi))
#grid_Hyperparameters = list(itertools.product(lr, batch_size, phi))

# gumbel_temp_min_list = [0.05, 0.2, 0.5]
# gumbel_temp_decay_list = [1e-4, 5e-5, 1e-5]
# gumbel_temp_update_every_list = [50, 100, 500]
# soft_rank_temp_list = [0.1, 0.3, 1.0]

gumbel_temp_min_list = [0.2, 0.5]
gumbel_temp_decay_list = [1e-4, 5e-5, 1e-5]
gumbel_temp_update_every_list = [100,500]
soft_rank_temp_list = [0.1]

grid_Hyperparameters = list(itertools.product(
    gumbel_temp_min_list,
    gumbel_temp_decay_list,
    gumbel_temp_update_every_list,
    soft_rank_temp_list
))



# Run --------------------



def run_experiment(index, combo):

    #lr, batch_size, y_gate_temp = combo 
    #lr, batch_size, phi = combo 
    #phi = combo[0]
    gumbel_temp_min, gumbel_temp_decay, gumbel_temp_update_every, soft_rank_temp = combo

    #gpu_id = index % max(1, NUM_GPUS) 
    gpu_id = index % NUM_GPUS if USE_ALL_GPUS and NUM_GPUS > 0 else PREFERRED_GPU

    #exp_name = f"lr{lr}_bs{batch_size}_gbt{y_gate_temp}_nh{num_heads}_dw{num_hidden_unit_w}_dr{num_hidden_unit_r}_phi{phi:.0e}_psi{psi}" # Create experiment name
    exp_name = f"TopYears35_Tm{gumbel_temp_min}_De{gumbel_temp_decay}_Up{gumbel_temp_update_every}_So{soft_rank_temp}".replace(".", "o")
    base_dir = pathlib.Path(save_dir) / exp_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Train command
    train_cmd = [
        "python", train_script,
        "--experiment_title", exp_name,
        "--save_dir", save_dir,
        "--device", f"cuda:{gpu_id}",
    ]
    for k, v in fixed.items(): # Append fixed arguments
        train_cmd += [f"--{k}", str(v)]
    train_cmd += [ # Append grid-specific hyperparameters
        "--gumbel_temp_min", str(gumbel_temp_min),
        "--gumbel_temp_decay", str(gumbel_temp_decay),
        "--gumbel_temp_update_every", str(gumbel_temp_update_every),
        "--soft_rank_temp", str(soft_rank_temp),
    ]


    # Eval command
    eval_cmd = [
        "python", eval_script,
        "--experiment_title", exp_name,
        "--save_dir", save_dir,
        "--device", f"cuda:{gpu_id}",
        "--num_epochs", str(fixed["num_epochs"]),
        "--num_sim", str(fixed["num_sim"]),
        "--batch_size", str(fixed["batch_size"]),
        "--seed_eval", str(seed_eval),
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
