
from source.utils.imports import *
from source.economic import Economic
from source.utils.utils import loss_function
from source.model.model import Model

def train_step(model: Model, dataloader, epoch, s_writer, optimizer, device, args):
    
    model.train()
    train_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch", leave=False)

    for batch_idx, batch in enumerate(train_iterator):
        
        if epoch == 1:
            if batch_idx == 5:
                pass

        theta_t_, w_t_, edu_ = batch
        w_t_ = w_t_.to(device)

        len_batch = len(batch[0])
        
        a_1 = torch.tensor([Economic.A_1]* len_batch, dtype=torch.float32)

        
        optimizer.zero_grad()
        all_a, all_c, all_c_ER, all_pr_bar, all_pr, all_h, all_y = model(theta_t_.to(device), edu_.to(device), a_1.to(device),w_t_)
        
        
        # if (all_c[:, ] <= 0).any():
        #     print('dfdf')
        if (all_c_ER <= 0).any():
            print('dfdf')
        
        
        
        # print('model phi', model.phi)
        # print('model psi', model.psi)


        # nan_gradients = False
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"Gradient of parameter '{name}' contains NaN")
        #         nan_gradients = True


        loss = loss_function(model, all_c, all_c_ER, all_pr_bar, all_pr, all_h, epoch, s_writer, args)


        
        s_writer.add_scalar('Loss/all', loss.item(), epoch)

        loss.backward()
        optimizer.step()
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"Gradient of parameter '{name}' contains NaN")
        #         nan_gradients = True
        
        with torch.no_grad():
            model.phi.clamp_(min=1e-5)
            model.psi.clamp_(min=1e-8)
            
            
        

        # # loss.detach().cpu()
        # print('model phi grad', model.phi.grad)
        # print('model psi grad', model.psi.grad)
        train_iterator.set_postfix(loss=loss.item())



  
  
  

def evaluation(model, dataloader, device):
  
    
    
    all_w = []
    all_a = []
    all_h = []
    all_y = []
    all_c = []
    all_p = []

    all_edu = []
    all_theta = []
    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():

            theta_t, w_t, edu = batch
            
            w_t = w_t.to(device)
            len_batch = len(batch[0])
            a_1 = torch.tensor([Economic.A_1]* len_batch)
            
            a_t, c_t_e, all_c_ER, pr, all_pr, h_t, y_t  = model(theta_t.to(device), edu.to(device), a_1.to(device),w_t)
            all_c_r = (1-pr) * ( all_pr * all_c_ER[:,:,1] + (1-all_pr) * all_c_ER[:,:,0]) +  pr * all_c_ER[:,:,2]
            
            c_t = torch.concat([c_t_e[:, :Economic.T_ER - Economic.AGE_0], all_c_r,c_t_e[:, Economic.T_LR - Economic.AGE_0+1:] ], dim = -1)
        

            all_w.extend(w_t.cpu())
            all_edu.extend(edu.cpu())
            
            all_theta.extend(theta_t.cpu())
            
            all_a.extend(a_t.cpu())
            all_h.extend(h_t.cpu())
            all_y.extend(y_t.cpu())
            all_c.extend(c_t.cpu())
            all_p.extend(pr.cpu())
            # all_t.extend(r_t.cpu())
            
            
    all_w = torch.stack(all_w)
    all_a = torch.stack(all_a)
    all_h = torch.stack(all_h)
    all_y = torch.stack(all_y)  
    all_c = torch.stack(all_c)
    all_theta = torch.stack(all_theta)
    all_p =  torch.stack(all_p)
    all_edu = torch.stack(all_edu)
    return   all_a, all_h, all_w, all_c, all_y, all_p, all_edu, all_theta
    

    
  