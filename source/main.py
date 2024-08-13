
from utils.imports import *
from utils.utils import save_checkpoint, generating_dataset
from model.model import Model
from args import get_parser
from economic import Economic
from train.train import train_step









def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    arg_dict = vars(args)
    
    




    #generating train dataset
    dataset_train = generating_dataset(Economic.J,  Economic.T_LR - Economic.AGE_0, Economic.THETA_0,Economic.P_EDU)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    
    
    
    
    if args.device == 'cuda':
        device = torch.device(f"cuda:{args.cuda_no}")
    elif args.device == 'cpu':
        device = 'cpu'
    num_epochs = args.num_epochs
    
    optimizer_func = optim.AdamW
    
    
    #if the input learning rate were 0 it means we want to check all the predefined learning rates
    if (args.lr == 0):
        lrs = [1e-1, 1e-2, 1e-3]
    else:
        lrs = [args.lr]
        
        
        
        
    for lr in lrs:
        
        
        #if the input num hidden units were 0 it means we want to check all the predefined nhu.
        if (args.num_hidden_unit_w == 0):
            num_h_u_w = [30,10]
        else:
            num_h_u_w = [args.num_hidden_unit_w]
            
            
            
            
            
        for num_h_u in num_h_u_w:
            
            model = Model(num_hidden_node_w=num_h_u, alpha_pr= args.alpha_pr , hard_gumbel=args.hard_gumbel)
            
            param_groups = [

                # {'params': model.phi, 'lr': args.lr_phi},
                # {'params': model.psi, 'lr': args.lr_psi},
                {'params': model.blocks.parameters()}
                
            ]
            optimizer = optimizer_func(param_groups, lr=lr)
            model.to(device)
            
            base_dir  = pathlib.Path(
                f'{args.save_dir}/{args.experiment_title}'\
                f'/base_model_with_regu_wb_{args.reg_mode}_{num_h_u}HiddenUnits_seed{args.seed}_phi{args.phi}'\
                f'/{args.batch_size}_batch_size'\
                f'/PSI{args.psi}'\
                f'/lambda{args.lmbd}'\
                f'/{optimizer_func.__name__}_lr:{lr}'
            )
                
            saved_model_dir = base_dir / "model"
            saved_plot_dir =  base_dir / "plot"
            saved_run_dir = base_dir / "runs"
                
            base_dir.mkdir(parents=True, exist_ok=True)
            saved_model_dir.mkdir(parents=True, exist_ok=True)
            saved_plot_dir.mkdir(parents=True, exist_ok=True)
            
            args_dict = vars(args)
            with open(f'{base_dir}/arguments.json', 'w') as file:
                json.dump(args_dict, file, indent=4)
            
            writer = SummaryWriter(saved_run_dir)
            
            
            for epoch in range(num_epochs):
                
                # if epoch == 400:
                #     optimizer.add_param_group({'params': model.phi, 'lr': args.lr_phi})
                #     optimizer.add_param_group({'params': model.psi, 'lr': args.lr_psi})
                print(epoch)
                train_step(model, dataloader_train, epoch, writer, optimizer, device, args)
                
                if (epoch%args.save_interval_epoch)==0:
                    
                    save_checkpoint(model, optimizer, saved_model_dir, epoch)

            save_checkpoint(model, optimizer, saved_model_dir, epoch)





if __name__ == "__main__":
   
    parser = get_parser()
    args = parser.parse_args()
    main(args)