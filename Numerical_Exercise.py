
import subprocess
import pathlib


# Configuration --------------------


# Arguments
hyperparams = {

    # Configuration
    #"experiment_title": "Numerical_Exercise_SQ_Nosigmoid_CPU_Uofh01", HighGenLayR _NoBaNorR  _LayerNormT _tlPlain _dr25
    "experiment_title": "Numerical_Exercise_LifeTimes_LS_Norout_MidEpoch_MidNS_HidW4", # _HidW8_HidR3 _dr25 _HidW4 _HidW4 _FreRes_VVLfre
    "save_dir": "./Experiments",
    "device": "cuda:1",
    #"device": "cpu",
    # Economic Enviernment
    "psi": 0.001, #0.00001,
    "phi": 0.001, #0.001,
    # Policy
    # "SS_Type": "Top_Years",
    # "SS_Param": 35,
    "SS_Type": "Life_Time",
    "SS_Param": 0,
    # DNN 
    "num_heads": 1, #2,
    "num_hidden_unit_w": 4, #16,
    "num_hidden_unit_r": 3, #5,                                                           
    "y_gate_temp": 5,
    "dropout": 0.00,
    # Optimization
    "lr": 5e-4, #1e-3,
    "batch_size": 512, #1024,
    "num_epochs": 250, #250,
    # Transfer LEarning
    "transfer_learning_type": "none",  # none
    "pretrained_path": "./Experiments/Numerical_Exercise_LifeTimes_NoLS_LowHidLay_HighwayR_MidEpoch_Highlr/model/epoch100/model.pt",
    "freeze_epochs": 60,             # optional if relevant
    "warm_epochs": 50,               # optional if relevant
    "warm_coef": 0.01,               # optional if relevant
    "freeze_res_epochs": 10,        # optional if relevant
    # Simulation
    "num_sim": 50000, #100000,
    # Randomness
    "seed_train": 92,
    "seed_eval": 776

}

# Paths to scripts
train_script = "Train.py"
eval_script = "Eval.py"

# Directory setup
base_dir = pathlib.Path(hyperparams["save_dir"]) / hyperparams["experiment_title"]
base_dir.mkdir(parents=True, exist_ok=True)


# Run Training --------------------


train_cmd = [
    "python", train_script,
    "--experiment_title", hyperparams["experiment_title"],
    "--save_dir", hyperparams["save_dir"],
    "--lr", str(hyperparams["lr"]),
    "--num_sim", str(hyperparams["num_sim"]), 
    "--batch_size", str(hyperparams["batch_size"]),
    "--num_epochs", str(hyperparams["num_epochs"]),
    "--device", hyperparams["device"],
    "--num_hidden_unit_w", str(hyperparams["num_hidden_unit_w"]),
    "--num_hidden_unit_r", str(hyperparams["num_hidden_unit_r"]),
    "--SS_Type", hyperparams["SS_Type"],
    "--SS_Param", str(hyperparams["SS_Param"]),
    "--dropout", str(hyperparams["dropout"]),
    "--psi", str(hyperparams["psi"]),
    "--phi", str(hyperparams["phi"]),
    "--seed_train", str(hyperparams["seed_train"]),
    "--transfer_learning_type", hyperparams["transfer_learning_type"], 
    "--pretrained_path", hyperparams["pretrained_path"], 
    "--freeze_epochs", str(hyperparams["freeze_epochs"]),           # optional if relevant       
    "--warm_epochs", str(hyperparams["warm_epochs"]),               # optional if relevant
    "--warm_coef", str(hyperparams["warm_coef"]),                   # optional if relevant
    "--freeze_res_epochs", str(hyperparams["freeze_res_epochs"]),   # optional if relevant
]

print(f"🚀 Starting training for experiment: {hyperparams['experiment_title']}")
with open(base_dir / "train.log", "w") as fout:
    subprocess.run(train_cmd, stdout=fout, stderr=subprocess.STDOUT, check=True)
print("✅ Finished training.")


# Run Evlauation --------------------


eval_cmd = [
    "python", eval_script,
    "--experiment_title", hyperparams["experiment_title"],
    "--save_dir", hyperparams["save_dir"],
    "--device", hyperparams["device"],
    "--num_epochs", str(hyperparams["num_epochs"]),
    "--num_sim", str(hyperparams["num_sim"]), 
    "--batch_size", str(hyperparams["batch_size"]),
    #"--num_hidden_unit_w", str(hyperparams["num_hidden_unit_w"]),
    #"--num_hidden_unit_r", str(hyperparams["num_hidden_unit_r"]),
    #"--SS_Type", hyperparams["SS_Type"],
    #"--SS_Param", str(hyperparams["SS_Param"]),
    #"--psi", str(hyperparams["psi"]),
    #"--phi", str(hyperparams["phi"]),
    "--seed_eval", str(hyperparams["seed_eval"])
]

print(f"🚀 Starting evaluation for experiment: {hyperparams['experiment_title']}")
with open(base_dir / "eval.log", "w") as fout:
    subprocess.run(eval_cmd, stdout=fout, stderr=subprocess.STDOUT, check=True)
print("✅ Finished evaluation.")