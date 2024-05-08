
from imports import *

from Parameters import *
from functions import *
from utils import *
from model import Model









def train_step(model, dataloader, epoch, s_writer, optimizer, device, args):
    
    model.train()
    train_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch", leave=False)

    for batch_idx, batch in enumerate(train_iterator):

        theta_t_, w_t_, edu_ = batch
        w_t_ = w_t_.to(device)

        len_batch = len(batch[0])
        
        a_1 = torch.tensor([A_1]* len_batch, dtype=torch.float32)

        
        optimizer.zero_grad()
        all_a, all_c, all_pr_bar, all_h, all_y = model(theta_t_.to(device), edu_.to(device), a_1.to(device),w_t_)
        
        # c_t = torch.concat([c_w, (1-pr) * c_wr[:, :, 0] + pr * c_wr[:,:, 1], c_r ], dim = -1)

        loss = loss_function_retirement_pr_cross(model, all_c,  all_h, epoch, s_writer, args)
        # print(epoch, batch_idx, loss.item())
        
        s_writer.add_scalar('Loss/all', loss.item(), epoch)
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        loss.detach().cpu()
        train_iterator.set_postfix(loss=loss.item())




def do_eval_save(model, dataloader, base_dir, epoch, device, s_writer,args):
    
    all_w = []
    all_a = []
    all_h = []
    all_y = []
    all_c = []
    all_t = []
    all_edu = []
    all_theta = []
    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():

            theta_t, w_t, edu = batch
            
            w_t = w_t.to(device)
            len_batch = len(batch[0])
            a_1 = torch.tensor([A_1]* len_batch)
            

            retirement_p, working_c, retirement_c, working_a, retirement_a, working_h, retirement_h, all_y_, all_r  = model(theta_t.to(device), edu.to(device), a_1.to(device), w_t)
            L = retirement_c.shape[1]
            
            # r_t = (torch.concat([all_r>.5, torch.ones(len_batch, 1).to(device)], dim=-1)).argmax(dim=-1)

            
            a_t = torch.concat([working_a, retirement_a.transpose(1,2)[torch.arange(len_batch), r_t]], dim =1)
            c_t = torch.concat([working_c, retirement_c.transpose(1,2)[torch.arange(len_batch), r_t]], dim =1)
            h_t = torch.concat([working_h, retirement_h.transpose(1,2)[torch.arange(len_batch), r_t]], dim =1)
            
            
            
            all_theta.extend(theta_t.cpu())
            all_w.extend(w_t.cpu())
            all_edu.extend(edu.cpu())
            
            all_a.extend(a_t.cpu())
            all_h.extend(h_t.cpu())
            all_y.extend(all_y_.cpu())
            all_c.extend(c_t.cpu())
            all_t.extend(r_t.cpu())
            
            
    all_w = torch.stack(all_w)
    all_a = torch.stack(all_a)
    all_h = torch.stack(all_h)
    all_y = torch.stack(all_y)  
    all_c = torch.stack(all_c)
    all_theta = torch.stack(all_theta)  
    
    # utility_retirement(all_c, all_h, epoch, s_writer,args, mode='eval')
    draw_all_plots(base_dir, all_a, all_h, all_w, all_theta, all_c, all_y, epoch)
    
    
    
  
  
  
  
def main(args):
  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    arg_dict = vars(args)
    
    




    #generating train dataset
    dataset_train = generating_dataset(J,  T_LR-AGE_0, THETA_0, P_EDU)
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
            
            model = Model(num_hidden_node_w=num_h_u, alpha_pr= args.alpha_pr )
            optimizer = optimizer_func(model.parameters(), lr=lr)
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
                # print(epoch)
                train_step(model, dataloader_train, epoch, writer, optimizer, device, args)
                
                if (epoch%args.save_interval_epoch)==0:
                    
                    save_checkpoint(model, optimizer, saved_model_dir, epoch)
                    # torch.save(model, saved_model_dir/f'model_epoch{epoch}.pt')

            # torch.save(model, f'{base_dir}/model.pt')        
            # do_eval_save(model, dataloader_eval, base_dir, epoch, device, writer, args)
            save_checkpoint(model, optimizer, saved_model_dir, epoch)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    
    # Step 3: Add arguments
    # Saved_stuff/multipleBlockR--pr--weighted_sum_H/
    parser.add_argument('--experiment_title', type=str)
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=92,  help='random seed for reproducibility')
    parser.add_argument('--num_hidden_unit_w', default=10, type=int, help='number of hidden units for the working network')
    parser.add_argument('--num_hidden_unit_r', default=5, type=int, help='number of hidden units for the retirement network')
    parser.add_argument('--num_epochs', type=int, default=1000,  help='number of epochs')
    parser.add_argument('--reg_mode', type=str, choices=['each10', 'last_year'],default='each10', help='Choose a regularization mode (each10,last_year)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lmbd', type=float, default=1e-2, help='retirement lambda')
    parser.add_argument('--psi', type=float, default=0.04, help='work hour disutility coefficient')
    parser.add_argument('--save_interval_epoch', type=int, default=100, help='Number of epochs between saving model checkpoints during training')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda_no', type=int, choices=[0,1],default=0, help='the index of cuda device to run the code on it')
    parser.add_argument('--save_dir', type=str,default='./Experiments', help='the directory that the training result will be saved in')
    parser.add_argument('--phi', type=float, default=0.0006, help='Phi in utility')
    parser.add_argument('--alpha_pr', type=float, default=5, help='the slop of sigmoid funciotn of pr')
 
    
    args = parser.parse_args()
    main(args)