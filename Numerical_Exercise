# SQ Model  --------------------


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
