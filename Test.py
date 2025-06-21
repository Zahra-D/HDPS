
import subprocess
import pathlib


# Configuration --------------------


# Arguments
hyperparams = {
    "lr": 1e-3,
    "batch_size": 1024,
    "experiment_title": "Numerical_Exercise",
    "save_dir": "./Experiments",
    "num_sim": 100000,
    "num_epochs": 50,
    "device": "cuda:0",
    "psi": 0.01,
    "phi": 0.0006,
    "num_heads": 2,
    "num_hidden_unit_w": 16,
    "num_hidden_unit_r": 5,
    "SS_Type": "Top_Years",
    "SS_Param": 35,
    "dropout": 0.0,
    "seed_train": 92,
    "seed_eval": 776
}

# Paths to scripts
train_script = "Train.py"
eval_script = "Eval.py"

# Directory setup
exp_dir = pathlib.Path(hyperparams["save_dir"]) / hyperparams["experiment_title"]
exp_dir.mkdir(parents=True, exist_ok=True)


# Run Training --------------------


# train_cmd = [
#     "python", train_script,
#     "--experiment_title", hyperparams["experiment_title"],
#     "--lr", str(hyperparams["lr"]),
#     "--num_sim", str(hyperparams["num_sim"]), 
#     "--batch_size", str(hyperparams["batch_size"]),
#     "--save_dir", hyperparams["save_dir"],
#     "--num_epochs", str(hyperparams["num_epochs"]),
#     "--device", hyperparams["device"],
#     "--num_hidden_unit_w", str(hyperparams["num_hidden_unit_w"]),
#     "--num_hidden_unit_r", str(hyperparams["num_hidden_unit_r"]),
#     "--SS_Type", hyperparams["SS_Type"],
#     "--SS_Param", str(hyperparams["SS_Param"]),
#     "--dropout", str(hyperparams["dropout"]),
#     "--psi", str(hyperparams["psi"]),
#     "--phi", str(hyperparams["phi"]),
#     "--seed_train", str(hyperparams["seed_train"])
# ]
# print(f"🚀 Starting training for experiment: {hyperparams['experiment_title']}")
# subprocess.run(train_cmd, check=True)
# print("✅ Finished training.")


# Run Evlauation --------------------


eval_cmd = [
    "python", eval_script,
    "--experiment_title", hyperparams["experiment_title"],
    "--num_sim", str(hyperparams["num_sim"]), 
    "--num_epochs", str(hyperparams["num_epochs"]),
    "--base_dir", hyperparams["save_dir"],
    "--device", hyperparams["device"],
    "--batch_size", str(hyperparams["batch_size"]),
    "--seed_eval", str(hyperparams["seed_eval"])
]
print(f"🚀 Starting evaluation for experiment: {hyperparams['experiment_title']}")
subprocess.run(eval_cmd, check=True)
print("✅ Finished evaluation.")