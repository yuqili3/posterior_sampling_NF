# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 02:45:46 2022

@author: liyuq
"""
# Import pytorch libraries
import torch
import torch.nn as nn

# Import ETC
import argparse, pprint, json
from types import SimpleNamespace
import copy, os
import time, datetime 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


import synthetic_dataset_LD  as data_LD
from model_hcnaf import MaskedWeight, Tanh_HCNAF, HyperNN_L2s, conditional_AF_layer, Sequential_HCNAF
import utils


def build_HCNAF_model(dataset, device):
    # Create HCNAF model
    """
        Create a HCNAF model; (1) Hyper-network (2) Conditional Autoregressive Flow
    """
    # Define a hyper-network
    n_layers_flow = 3
    dim_h_flow = 64
    norm_HW = 'scaled_frobenius'
    HyperLayer = HyperNN_L2s(dataset, n_layers_flow = n_layers_flow, dim_h_flow = dim_h_flow, NSP=True, device=device)
    
    # Define a conditional AF
    intermediate_layers_cAFs = []
    dim_o = dataset.n - dataset.m
    
    for _ in range(n_layers_flow - 1):
        intermediate_layers_cAFs.append(MaskedWeight(dim_o * dim_h_flow, dim_o * dim_h_flow, dim=dim_o, norm_w=norm_HW))
        intermediate_layers_cAFs.append(Tanh_HCNAF())

    conditional_AFs = conditional_AF_layer(
        *([MaskedWeight(dim_o, dim_o * dim_h_flow, dim=dim_o, norm_w=norm_HW), Tanh_HCNAF()] + \
        intermediate_layers_cAFs + \
        [MaskedWeight(dim_o * dim_h_flow, dim_o, dim=dim_o, norm_w=norm_HW)]))

    model = Sequential_HCNAF(HyperLayer, conditional_AFs).to(device)
    
    # print('{}'.format(model))
    print('# of parameters={}'.format(sum((param != 0).sum() for param in model.parameters())))

    return model

def train_HCNAF_cnf(cnf, dataset_name, num_samples=1000, noise_sigma=0.0, batch_size=1000):
    loss_min = float('inf')
    
    cnf.train()
    
    optimizer = torch.optim.Adam(cnf.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-5, verbose=True, threshold_mode='abs')
    
    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    # Training parameters
    n_epochs = args.iterations
    t = trange(n_epochs)
    for i_epoch in t:
        l_tot = 0
        batch_idx = 0
        x, y = dataset.generate_dataset(num_samples=num_samples, noise_sigma=noise_sigma)
        train_loader = utils.get_dataloader(dataset, batch_size=batch_size)
    
        
        for x, y in train_loader:
            y = y.reshape(-1, 1)
            x_null = x[:, dataset.m:]
            batch_idx += 1
            z, log_det_j, HyperParam = cnf(torch.cat((y, x_null), dim=1))
            log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
            loss = (-log_p_z - log_det_j).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), max_norm=0.1)
        
            optimizer.step()
            optimizer.zero_grad()
            l_tot += loss.data.item()
            
        cur_loss = l_tot / batch_idx
        t.set_description('training loss=%g' % (cur_loss))
        scheduler.step(cur_loss)
        
        if i_epoch%args.savemodel_period == 0 and i_epoch > 0:
            
            if cur_loss < loss_min:
                loss_min = cur_loss
                best_model_state = cnf.state_dict()
                print('Saving progress, saving the current best model, at iteration {}, best loss {}'.format(i_epoch, loss_min))
                
            save_name = '%s_noise_%.4e_f_checkpoint_nsp'%(dataset_name, noise_sigma)
            save_dict = {
                'save_path': args.path + save_name,
                'best_iteration': i_epoch,
                'best_loss': loss_min,
                'best_model_state_dict': best_model_state,
                'cur_model_state_dict': cnf.state_dict(),
            }
            utils.save_state(save_dict)
            # print & save loss & val_loss
            perf_eval = 'Iteration [{}/{}], cur_training_Loss: {:.4f}, training_Loss_min: {:.4f}'.format(i_epoch, n_epochs, cur_loss, loss_min)
            print(perf_eval, file=open(args.path + '%s_noise_%.4e_f_losses_nsp.txt'%(dataset_name, noise_sigma), "a"))


def train_HCNAF_generator(cnf, generator, dataset_name, num_samples=1000, noise_sigma=0.0, batch_size=1000): 
    loss_min = float('inf')
    
    cnf.eval()
    generator.train()
    
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-5, verbose=True, threshold_mode='abs')
    mse = nn.MSELoss()
    
    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    # Training parameters
    n_epochs = args.iterations
    t = trange(n_epochs)
    for i_epoch in t:
        x, y = dataset.generate_dataset(num_samples=num_samples, noise_sigma=noise_sigma)
        train_loader = utils.get_dataloader(dataset, batch_size=batch_size)
    
        l_tot = 0
        batch_idx = 0
        
        for x, y in train_loader:
            y = y.reshape(-1,1)
            x_null = x[:, dataset.m:]
            batch_idx += 1
            
            z, log_det_j, Hyperparam = cnf(torch.cat((y, x_null), dim=1))
            x_, _, _ = generator(torch.cat((y, z), dim=1))
            
            loss = mse(x_null, x_).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.1)
    
            optimizer.step()
            optimizer.zero_grad()
            l_tot += loss.data.item()
            
        cur_loss = l_tot / batch_idx
        t.set_description('training loss=%g' % (cur_loss))
        scheduler.step(cur_loss)
        
        if i_epoch%args.savemodel_period == 0 and i_epoch > 0:
            
            if cur_loss < loss_min:
                loss_min = cur_loss
                best_model_state = generator.state_dict()
                print('Saving progress, saving the current best model, at iteration {}, best loss {}'.format(i_epoch, loss_min))
                
            save_name = '%s_noise_%.4e_g_checkpoint_nsp'%(dataset_name, noise_sigma)
            save_dict = {
                'save_path': args.path + save_name,
                'best_iteration': i_epoch,
                'best_loss': loss_min,
                'best_model_state_dict': best_model_state,
                'cur_model_state_dict': generator.state_dict(),
            }
            utils.save_state(save_dict)
            # print & save loss & val_loss
            perf_eval = 'Iteration [{}/{}], cur_training_Loss: {:.4e}, training_Loss_min: {:.4e}'.format(i_epoch, n_epochs, cur_loss, loss_min)
            print(perf_eval, file=open(args.path + '%s_noise_%.4e_g_losses_nsp.txt'%(dataset_name, noise_sigma), "a"))
     
        
     
# Computing Loss function
def compute_log_p_x(model, data):
    z, log_det_j, HyperParam = model(data)
    log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
#    print('log_p_z: ', log_p_z, 'log_det_j: ', log_det_j)
    return log_p_z + log_det_j

def test_HCNAF_density(cnf, dataset_name, noise_sigma=0.0):
    
    cnf.eval()
    
    '''
    n = 20
    y0 = np.random.rand()
    x0 = np.sqrt(1-y0**2)
    print(y0, x0)
    y = y0 * torch.ones((n, 1))
    unit_vec = torch.randn(n, 2)
    unit_vec /= torch.norm(unit_vec, dim=1, keepdim=True)
    
    y_x = torch.cat((y, x0 * unit_vec), dim=1)
    p = compute_log_p_x(cnf, y_x.to(device))
    print(p)
    
    unit_vec = torch.randn(n, 2)
    unit_vec /= torch.norm(unit_vec, dim=1, keepdim=True)
    unit_vec += 0.01/x0 *torch.randn_like(unit_vec)
    y_x = torch.cat((y, x0 * unit_vec), dim=1)
    p = compute_log_p_x(cnf, y_x.to(device))
    print(p)
    
    y_x = torch.cat((y, 2*torch.randn(n, 2)), dim=1)
    p = compute_log_p_x(cnf, y_x.to(device))
    print(p)
    '''

    dataset = data_LD.synth_dataset_LD(dataset_name)
    if dataset_name == 'circle':
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = -1.5, -1.5, 1.5, 1.5
    elif dataset_name == 'single_sphere':
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = -1.5, -1.5, 1.5, 1.5
    elif dataset_name == 'concentric_spheres':
        lim_x_l, lim_x_u= -dataset.number_of_spheres-0.5, dataset.number_of_spheres+0.5
        lim_y_l, lim_y_u= -dataset.number_of_spheres-0.5, dataset.number_of_spheres+0.5
    elif dataset_name == 'intersect_spheres':
        lim_x_l, lim_y_l, lim_x_u, lim_y_u = -2, -1.5, 2, 1.5
        
    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    if dataset_name == 'concentric_sphers':
        y_list = np.arange(0, 1.9, 0.2) 
    else:
        y_list = np.arange(0, 1.01, 0.1) 

    fig, axes = plt.subplots(2, len(y_list), figsize=(4*len(y_list), 8))
    
    step = 0.1
    
    for i, cond_y in enumerate(y_list):
        x_grid = np.array([[a, b] for b in np.arange(lim_x_l, lim_x_u, step) for a in np.arange(lim_y_l, lim_y_u, step)])
        dataset.x = x_grid
        dataset.y = cond_y * np.ones(len(x_grid)).reshape(-1, 1)
        grid_dataloader = utils.get_dataloader(dataset, shuffle=False)
        prob = torch.cat([torch.exp(compute_log_p_x(cnf, torch.cat((y,x), dim=1))).detach() for x, y in grid_dataloader], 0)
        prob = prob.view(int((lim_x_u - lim_x_l) / step), int((lim_y_u - lim_y_l) / step))
        axes[0, i].grid('on')
        axes[0, i].imshow(prob.cpu().data.numpy(), extent=(lim_x_l, lim_x_u, lim_y_l, lim_y_u), cmap='Reds')
        axes[0, i].set_title('$\widehat{P}(x|y=%.2f)$'%(cond_y))
        
    fig.tight_layout()
    fig.savefig('%s%s_noise_%.4e_density_nsp.png'%(args.fig_path, args.dataset_name, args.noise_sigma))
    
    
    
    
def test_HCNAF_generate(cnf, generator, dataset_name,  noise_sigma = 0.0):
    cnf.eval()
    generator.eval()

    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    ndim_x = dataset.n
    ndim_y = dataset.m
    ndim_z = dataset.n - dataset.m
    
    if dataset_name == 'concentric_sphers':
        y_list = np.arange(0, 2.01, 0.1) 
    else:
        y_list = np.arange(0, 1.01, 0.1) 
        
    fig, axes = plt.subplots(2, len(y_list), figsize=(4*len(y_list), 8))
    
    test_loaders = []
    xs = []
    ys = []
    display_nums = 5000
    
    
    for i, cond_y in enumerate(y_list):
        x, y = dataset.generate_dataset(num_samples=display_nums, noise_sigma=noise_sigma, condition_y =  cond_y* np.ones(display_nums))
        test_loader = utils.get_dataloader(dataset,batch_size=display_nums)
        test_loaders.append(test_loader)
        xs.append(x)
        ys.append(y)
    
    for i in range(len(test_loaders)):
        x_samps = torch.cat([x for x,y in test_loaders[i]], dim=0)[:display_nums]
        x_null = x_samps[:, dataset.m:]
        y_samps = torch.cat([y for x,y in test_loaders[i]], dim=0)[:display_nums].reshape(-1,1)
        
        z_true, _, _ = cnf(torch.cat((y_samps, x_null), dim=1).to(args.device))
        x_null_, _, _  = generator(torch.cat((y_samps, z_true), dim=1).to(args.device))
        
        # x_null_, _, _  = generator(torch.cat((y_samps, torch.randn(display_nums, ndim_z).to(args.device)), dim=1))
        
        print(torch.norm(x_null_, dim=1), x_null[:,0] / x_null[:,1])
        
        x_null_ = x_null_.cpu().data.numpy()
        
        axes[0, i].set_title('$x\sim p^*(x|y=%.2f)$'%(y_list[i]))
        dataset.display_dataset(axes[0, i], xs[i], ys[i], display_nums=display_nums, color='r')
        
        axes[1, i].set_title('$g(y=%.2f, z), z\sim N(0,I)$'%(y_list[i]))
        dataset.display_dataset(axes[1, i], np.c_[ys[i], x_null_], ys[i], display_nums=display_nums)
        
    fig.tight_layout()
    fig.savefig('%s%s_noise_%.4e_real_generated_samples_nsp.png'%(args.fig_path, args.dataset_name, args.noise_sigma))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HCNAF experiment")
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--dataset_name', type=str, default='single_sphere', choices=[ 'intersect_spheres', 'circle', 'concentric_spheres', 'single_sphere'])
    parser.add_argument('--noise_sigma', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=int(1e5))
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--savemodel_period', type=int, default=10)
    parser.add_argument('--train_f', type=int, default=0)
    parser.add_argument('--train_g', type=int, default=0)
    parser.add_argument('--test_f', type=int, default=0)
    parser.add_argument('--test_g', type=int, default=0)
    
    args = parser.parse_args()
    
    if not os.path.exists('synthetic_dataset/HCNAF'):
        os.makedirs('synthetic_dataset/HCNAF')
    if not os.path.exists('synthetic_dataset/HCNAF_figures'):
        os.makedirs('synthetic_dataset/HCNAF_figures')
    args.path = 'synthetic_dataset/HCNAF/'
    args.fig_path = 'synthetic_dataset/HCNAF_figures/'
    
    dataset_name = args.dataset_name
    num_samples = args.num_samples
    noise_sigma = args.noise_sigma
    device = args.device
    
    batch_size = 1000
    
    dataset = data_LD.synth_dataset_LD(dataset_name)
    cnf = build_HCNAF_model(dataset, device)
    generator = build_HCNAF_model(dataset, device)
    
    if args.train_f > 0:
        train_HCNAF_cnf(cnf, dataset_name, num_samples=num_samples, noise_sigma=noise_sigma, batch_size=batch_size)
        
    if args.train_g > 0:
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint_nsp.pt'%(dataset_name, noise_sigma), load_best =True)
        cnf.load_state_dict(best_model_state)
        train_HCNAF_generator(cnf, generator, dataset_name, num_samples=num_samples, noise_sigma=noise_sigma, batch_size=batch_size)
        
    if args.test_f > 0:
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint_nsp.pt'%(dataset_name, noise_sigma), load_best=True)
        cnf.load_state_dict(best_model_state)
        test_HCNAF_density(cnf, dataset_name, noise_sigma=noise_sigma)
        
    if args.test_g > 0:
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint_nsp.pt'%(dataset_name, noise_sigma), load_best=True)
        cnf.load_state_dict(best_model_state)
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_g_checkpoint_nsp.pt'%(dataset_name, noise_sigma), load_best=True)
        generator.load_state_dict(best_model_state)
        test_HCNAF_generate(cnf, generator, dataset_name, noise_sigma=noise_sigma)
        
        