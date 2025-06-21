
1. Paraemters: Parameters initialization

Defining paremters of 
- Life Cycle, 
- Initial Distribution, Wage Function, Preference, Budget Constaint,
- Grids, Simulation
- Tax Function , Social Security Function 

--------------------------------------------------------------------------------

2. Functions: Defining functions used in the model

- Wage Function
- Budget Constaint (consumtpion) 
- Public Policies (income tax, socail security)

--------------------------------------------------------------------------------

3. Model: Defninng the DNN model

- Working Ages Block
- Retierement Block
- Early Retierment Block
- Full Model



--------------------------------------------------------------------------------

4. utils: 




--------------------------------------------------------------------------------

* How to run:


- Verify pip installation: 

pip list

- Navigate to your project folder:

cd ~/HDPS

- Create a virtual environment:

python3 -m venv tf_Comp

- Activate the virtual environment:

source tf_Comp/bin/activate

- Install the required packages:

pip install -r requirements.txt

- Verify installation:

pip list

- Run Your Python Code:

python Numerical_Exercise.py

- Lunch Tensoroard (Training Logs)

tensorboard --logdir Experiments/Numerical_Exercise/runs

http://localhost:6006/ (in local Browser)

- Model Evaluation


1. Paraemters: Parameters initialization

Defining paremters of 
- Life Cycle, 
- Initial Distribution, Wage Function, Preference, Budget Constaint,
- Grids, Simulation
- Tax Function , Social Security Function 

--------------------------------------------------------------------------------

2. Functions: Defining functions used in the model

- Wage Function
- Budget Constaint (consumtpion) 
- Public Policies (income tax, socail security)

--------------------------------------------------------------------------------

3. Model: Defninng the DNN model

- Working Ages Block
- Retierement Block
- Early Retierment Block
- Full Model



--------------------------------------------------------------------------------

4. utils: 




--------------------------------------------------------------------------------

* How to run:


- Verify pip installation: 

pip list

- Navigate to your project folder:

cd ~/HDPS

- Create a virtual environment:

python3 -m venv tf_Comp

- Activate the virtual environment:

source tf_Comp/bin/activate

- Install the required packages:

pip install -r requirements.txt

- Verify installation:

pip list

- Run Your Python Code:

python Numerical_Exercise.py
python Search_Hyperparameters.py

- Model Evaluation

(Lunch Tensoroard for Training Logs)

tensorboard --logdir Experiments/Numerical_Exercise_LifeTimes
tensorboard --logdir Experiments/Numerical_Exercise_SQ_Nosigmoid_CPU_ZeroPsi
tensorboard --logdir ./Experiments/Search_Hyperparameters
tensorboard --logdir ./Experiments/Numerical_Exercise_LifeTimes_NoLS
tensorboard --logdir ./Experiments/Numerical_Exercise_LifeTimes_NoLS_LowHidLay
tensorboard --logdir ./Experiments/Search_Hyperparameters_LifeTime_wLS
tensorboard --logdir ./Experiments/

http://localhost:6006/ (in local Browser)

tail -f global_grid_search.log

pkill -f python


- Optimal Hyperparemters

python Search_Hyperparameters.py

- Monitor Run on GPU(s)

watch -n 2 nvidia-smi
watch -n 5 'ls -lh ./Experiments/GridSearch_Hyperparameters/*/model/*'

tail -f grid_search_progress.log



