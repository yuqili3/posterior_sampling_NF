# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 01:47:00 2022

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

import synthetic_dataset_LD  as data_LD
from model_hcnaf import MaskedWeight, Tanh_HCNAF, HyperNN_L2s, conditional_AF_layer, Sequential_HCNAF
import utils
import model_inn as inn


def build_HCNAF_model(dataset, device):
    # Create HCNAF model
    """
        Create a HCNAF model; (1) Hyper-network (2) Conditional Autoregressive Flow
    """
    # Define a hyper-network
    n_layers_flow = 3
    dim_h_flow = 64
    norm_HW = 'scaled_frobenius'
    HyperLayer = HyperNN_L2s(dataset, n_layers_flow = n_layers_flow, dim_h_flow = dim_h_flow, device=device)
    
    # Define a conditional AF
    intermediate_layers_cAFs = []
    dim_o = dataset.n
    
    for _ in range(n_layers_flow - 1):
        intermediate_layers_cAFs.append(MaskedWeight(dim_o * dim_h_flow, dim_o * dim_h_flow, dim=dim_o, norm_w=norm_HW))
        intermediate_layers_cAFs.append(Tanh_HCNAF())

    conditional_AFs = conditional_AF_layer(
        *([MaskedWeight(dim_o, dim_o * dim_h_flow, dim=dim_o, norm_w=norm_HW), Tanh_HCNAF()] + \
        intermediate_layers_cAFs + \
        [MaskedWeight(dim_o * dim_h_flow, dim_o, dim=dim_o, norm_w=norm_HW)]))

    model = Sequential_HCNAF(HyperLayer, conditional_AFs).to(args.device)
    
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
        train_loader = utils.get_dataloader(dataset, batch_size=batch_size, device=args.device)
    
        
        for x, y in train_loader:
            y = y.reshape(-1, 1)
            batch_idx += 1
            z, log_det_j, HyperParam = cnf(torch.cat((y, x), dim=1))
            log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
            loss = (-log_p_z - log_det_j).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), max_norm=1)
        
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
                
            save_name = '%s_noise_%.4e_f_checkpoint'%(dataset_name, noise_sigma)
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
            print(perf_eval, file=open(args.path + '%s_noise_%.4e_f_losses.txt'%(dataset_name, noise_sigma), "a"))


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
        train_loader = utils.get_dataloader(dataset, batch_size=batch_size, device=args.device)
    
        l_tot = 0
        batch_idx = 0
        
        for x, y in train_loader:
            y = y.reshape(-1,1)
            batch_idx += 1
            
            z, log_det_j, Hyperparam = cnf(torch.cat((y, x), dim=1))
            x_, _, _ = generator(torch.cat((y, z), dim=1))
            
            loss = mse(x, x_).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)
    
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
                
            save_name = '%s_noise_%.4e_g_checkpoint'%(dataset_name, noise_sigma)
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
            print(perf_eval, file=open(args.path + '%s_noise_%.4e_g_losses.txt'%(dataset_name, noise_sigma), "a"))
     
        
     
# Computing Loss function
def compute_log_p_x(model, data):
    z, log_det_j, HyperParam = model(data)
    log_p_z = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
#    print('log_p_z: ', log_p_z, 'log_det_j: ', log_det_j)
    return log_p_z + log_det_j

def plot_density(ax, cnf, cond_y, dataset_name):
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
    step = 0.04
    if dataset_name == 'circle':
        x_grid = np.array([[cond_y, b] for b in np.arange(lim_x_l, lim_x_u, step)])
    else:
        x_grid = np.array([[cond_y, a, b] for b in np.arange(lim_y_l, lim_y_u, step) for a in np.arange(lim_x_l, lim_x_u, step)])
        if dataset_name == 'intersect_spheres':
            c1 = np.array([0,0.75,0])
            c2 = np.array([0,-0.75,0])
            # for i, x in enumerate(x_grid):
            #     if abs(np.linalg.norm(x - c1) - 1) < abs(np.linalg.norm(x - c2) - 1) and abs(np.linalg.norm(x - c1) - 1) < 1.5*step:
            #         x_grid[i] = (x - c1) /np.linalg.norm(x - c1) + c1 
            #     if abs(np.linalg.norm(x - c1) - 1) > abs(np.linalg.norm(x - c2) - 1) and abs(np.linalg.norm(x - c2) - 1) < 1.5*step:
            #         x_grid[i] = (x - c2) /np.linalg.norm(x - c2) + c2 
    
    dataset.x = x_grid
    dataset.y = cond_y * np.ones(len(x_grid)).reshape(-1, 1)
    grid_dataloader = utils.get_dataloader(dataset, shuffle=False, device=args.device, batch_size=len(x_grid))
    prob = torch.cat([(compute_log_p_x(cnf, torch.cat((y, x), dim=1))).detach() for x, y in grid_dataloader], 0)
    # print(torch.max(prob), torch.min(prob), torch.log(torch.max(prob)), torch.log( torch.min(prob)))

    if dataset_name == 'circle':
        prob = prob.view(int((lim_x_u - lim_x_l) / step))
        ax.grid('on')
        im = ax.plot(prob.cpu().data.numpy(), '.--')
        # ax.set_xlabel('$x_1$')
        # ax.set_ylabel('$\widehat{p}(x|y)$')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        
        prob = prob.view(int((lim_y_u - lim_y_l) / step), int((lim_x_u - lim_x_l) / step))
        prob = prob.cpu().data.numpy()
        ax.grid('on')
        im = ax.imshow((prob), extent=(lim_x_l, lim_x_u, lim_y_l, lim_y_u), cmap='plasma')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # ax.set_xlabel('$x_1$')
        # ax.set_ylabel('$x_2$')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
def test_HCNAF_density(cnf, dataset_name, noise_sigma=0.0):    
    cnf.eval()
    
    # n = 20
    # y0 = np.random.rand()
    # x0 = np.sqrt(1-y0**2)
    # print(y0, x0)
    # y = y0 * torch.ones((n, 1))
    # unit_vec = torch.randn(n, 2)
    # unit_vec /= torch.norm(unit_vec, dim=1, keepdim=True)
    
    # y_x = torch.cat((y, y, x0 * unit_vec), dim=1)
    # p = compute_log_p_x(cnf, y_x.to(device))
    # print(p)
    
    # unit_vec = torch.randn(n, 2)
    # unit_vec /= torch.norm(unit_vec, dim=1, keepdim=True)
    # unit_vec += 0.01/x0 *torch.randn_like(unit_vec)
    # y_x = torch.cat((y, y, x0 * unit_vec), dim=1)
    # p = compute_log_p_x(cnf, y_x.to(device))
    # print(p)
    
    # y_x = torch.cat((y, y, torch.zeros(n, 2) + 0.01 * torch.randn(n, 2)), dim=1)
    # p = compute_log_p_x(cnf, y_x.to(device))
    # print(p)
    
    if dataset_name == 'concentric_spheres':
        # y_list = np.arange(0, 1.9, 0.4) 
        y_list = [0.3, 0.9, 1.6]
    else:
        # y_list = np.arange(0, 1.01, 0.2)
        y_list = [0.3, 0.6, 0.95]
    fig, axes = plt.subplots(1, len(y_list), figsize=(4*len(y_list), 4))
    for i, cond_y in enumerate(y_list):
        axes[i].grid('on')
        plot_density(axes[i], cnf, cond_y, dataset_name)
        axes[i].set_title('$\widehat{p}_{CNF}(x|y=%.2f)$'%(cond_y))
        
    fig.tight_layout()
    fig.savefig('%s%s_noise_%.4e_density.png'%(args.fig_path, args.dataset_name, args.noise_sigma))
    
    
    
def test_HCNAF_generate(cnf, generator, dataset_name,  noise_sigma = 0.0, plot=True, plot_circle =True):
    cnf.eval()
    generator.eval()
    
    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    ndim_x = dataset.n
    ndim_y = dataset.m
    ndim_z = dataset.n
    
    if dataset_name == 'concentric_spheres':
        # y_list = np.arange(0, 1.99, 0.02) 
        y_list = [0.3, 0.9, 1.6]
    else:
        # y_list = np.arange(0, 1.01, 0.02) 
        y_list = [0.3, 0.6, 0.95]
        
    fig, axes = plt.subplots(2, len(y_list), figsize=(4*len(y_list), 8))
    
    Ly, Lpq = np.zeros(len(y_list)), np.zeros(len(y_list))
    display_nums = int(2e5)
    batch_size = 1000
    with torch.no_grad():
        for i in trange(len(y_list)):
            ly, lpq = [], []
            cond_y = y_list[i]
            x, y = dataset.generate_dataset(num_samples=display_nums, noise_sigma=noise_sigma, condition_y =  cond_y* np.ones(display_nums))
            test_loader = utils.get_dataloader(dataset,batch_size=batch_size, device=args.device, shuffle=False)
            for tl, (x, y) in enumerate(test_loader):
                y_samps = y.reshape(-1,1)
                # z, _, _ = cnf(torch.cat((y_samps, x_samps), dim=1))
                # x_generated, _, _  = generator(torch.cat((y_samps, z), dim=1))
                x_generated, _, _  = generator(torch.cat((y_samps, torch.randn(batch_size, ndim_z).to(device)), dim=1))
                y_generated = x_generated[:,0]
                ly.append(((y_generated - y_samps)**2).squeeze().mean().cpu().data.numpy())
                lpq.append(inn.MMD_multiscale(x_generated, x, args.device).cpu().data.numpy())
                
                if plot:
                    x_generated = x_generated.cpu().data.numpy()
                    y = y.cpu().data.numpy()
                    # print(np.linalg.norm(x_generated[:, 1:], axis=1))
                    
                    heatmap_grid = 40 if dataset_name =='circle' else 80
                    # axes[0, i].set_title('$p^*(P_{N(A)}(x)|y=%.2f)$'%(cond_y))
                    # dataset.display_dataset(axes[0, i], x, y, display_nums=display_nums, color='r', heatmap=True, heatmap_grid=heatmap_grid)
                    plot_density(axes[0, i], cnf, cond_y, dataset_name)
                    
                    if plot_circle and tl == 0:
                        dataset.display_dataset(axes[1, i], x_generated, y, display_nums=display_nums, heatmap=True, heatmap_grid=heatmap_grid, plot_circle=True)
                    elif dataset_name != 'circle':
                        dataset.display_dataset(axes[1, i], x_generated, y, display_nums=display_nums, heatmap=True, heatmap_grid=heatmap_grid, plot_circle=False)
                    
            Ly[i] = np.array(ly).mean()
            Lpq[i] = np.array(lpq).mean()
            
            if plot:
                title = '$\widehat{p}(x_1|y=%.2f)$'%(cond_y) if dataset_name == 'circle' else '$x\sim p^*(x|y=%.2f)$'%(cond_y)
                axes[0, i].set_title(title, fontsize=20)
                
                title =  '$\widehat{p}(g(y=%.2f, z)_1)$'%(cond_y) if dataset_name == 'circle' else '$g(y=%.2f, z), z\sim N(0,I)$'%(cond_y)
                axes[1, i].set_title(title, fontsize=20)
            
        if plot:
            fig.tight_layout()
            fig.savefig('%s%s_noise_%.4e_probability_heatmaps.png'%(args.fig_path, args.dataset_name, args.noise_sigma))
            
        np.savez('%s%s_noise_%.4e_errors_temp'%(args.fig_path, args.dataset_name, args.noise_sigma),
             Ly = Ly, Lpq=Lpq, y_list=y_list)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HCNAF experiment")
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda:0','cuda:1'])
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--dataset_name', type=str, default='concentric_spheres', choices=[ 'intersect_spheres', 'circle', 'concentric_spheres', 'single_sphere'])
    parser.add_argument('--noise_sigma', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=int(1e5))
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--savemodel_period', type=int, default=10)
    parser.add_argument('--train_f', type=int, default=0)
    parser.add_argument('--train_g', type=int, default=0)
    parser.add_argument('--test_f', type=int, default=0)
    parser.add_argument('--test_g', type=int, default=0)
    parser.add_argument('--resume_train', type=int, default=0)
    
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
        if args.resume_train > 0:
            _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint.pt'%(dataset_name, noise_sigma), load_best=True)
            cnf.load_state_dict(best_model_state)
        train_HCNAF_cnf(cnf, dataset_name, num_samples=num_samples, noise_sigma=noise_sigma, batch_size=batch_size)
        
    if args.train_g > 0:
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint.pt'%(dataset_name, noise_sigma), load_best=True)
        cnf.load_state_dict(best_model_state)
        if args.resume_train > 0:
            _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_g_checkpoint.pt'%(dataset_name, noise_sigma), load_best=True)
            generator.load_state_dict(best_model_state)
        train_HCNAF_generator(cnf, generator, dataset_name, num_samples=num_samples, noise_sigma=noise_sigma, batch_size=batch_size)
        
    if args.test_f > 0:
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint.pt'%(dataset_name, noise_sigma), load_best=True)
        cnf.load_state_dict(best_model_state)
        test_HCNAF_density(cnf, dataset_name, noise_sigma=noise_sigma)
        
    if args.test_g > 0:
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_f_checkpoint.pt'%(dataset_name, noise_sigma), load_best=True)
        cnf.load_state_dict(best_model_state)
        _, _, best_model_state = utils.load_state(args.path + '%s_noise_%.4e_g_checkpoint.pt'%(dataset_name, noise_sigma), load_best=True)
        generator.load_state_dict(best_model_state)
        test_HCNAF_generate(cnf, generator, dataset_name, noise_sigma=noise_sigma)
        
        