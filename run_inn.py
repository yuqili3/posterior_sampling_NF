# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:26:11 2022

@author: liyuq
"""
from time import time
import argparse
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm,  trange
import os

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

import synthetic_dataset_LD  as data_LD
import model_inn as inn
import utils 

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def build_INN_model(ndim_tot):
    nodes = [InputNode(ndim_tot, name='input')]
    for k in range(8):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':inn.subnet_fc, 'clamp':2.0},
                          name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                          PermuteRandom,
                          {'seed':k},
                          name=F'permute_{k}'))
    
    nodes.append(OutputNode(nodes[-1], name='output'))
    
    model = ReversibleGraphNet(nodes, verbose=False)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    for param in trainable_parameters:
        param.data = 0.05*torch.randn_like(param)
    model.to(device)
    
    return model



def train_INN_model(model, dataset_name, num_samples=1e4, noise_sigma = 0.0, batch_size=1000):
    loss_min = float('inf')
    
    model.train()
    
    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    # Training parameters
    n_epochs = args.iterations
    
    lr = 1e-3
    l2_reg = 2e-5
    
    y_noise_scale = 1e-1
    zeros_noise_scale = 5e-2
    
    ndim_x = dataset.n
    ndim_y = dataset.m
    ndim_z = dataset.n - dataset.m
    
    
    # relative weighting of losses:
    lambd_predict = 3.
    lambd_latent = 300.
    lambd_rev = 400. 
    
    pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)
    pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)
    
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9),
                                 eps=1e-6, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, min_lr=1e-5, verbose=True, threshold_mode='abs')
    
    
    loss_backward = inn.MMD_multiscale
    loss_latent = inn.MMD_multiscale
    loss_fit = inn.fit
    
    t = trange(n_epochs)
    for i_epoch in t:
        x, y = dataset.generate_dataset(num_samples=num_samples, noise_sigma=noise_sigma)
        train_loader = utils.get_dataloader(dataset, batch_size=batch_size, device=args.device)
    
        l_tot = 0
        batch_idx = 0
        # If MMD on x-space is present from the start, the model can get stuck.
        # Instead, ramp it up exponetially.  
        loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / n_epochs)))
        for x, y in train_loader:
            batch_idx += 1
            x, y = x.to(device), y.to(device)

            y_clean = y.clone().reshape(-1, ndim_y)
            
            pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                    ndim_x, device=device)
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                     ndim_y - ndim_z, device=device)
            
            y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

            x, y = (torch.cat((x, pad_x),  dim=1),
                    torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                              dim=1))
            
            
            optimizer.zero_grad()
            # Forward step:
            output, jacobian = model(x)

            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)
            Ly = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])
            l = Ly
            ## || f_y(x) - y ||_2^2  + ||f_0(x) - 0||_2^2
            output_block_grad = torch.cat((output[:, :ndim_z],
                                           output[:, -ndim_y:].data), dim=1)
            Lpq = lambd_latent * loss_latent(output_block_grad, y_short, device=args.device)
            l += Lpq
            ## MMD(f_{y,z}(x), [y, z])
            l_tot += l.data.item()
            l.backward()
    
            # Backward step:
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                     ndim_y - ndim_z, device=device)
            y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)
            
            orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                                torch.randn(batch_size, ndim_z, device=device))
            y_rev = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)
            y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y), dim=1)
            
            output_rev, jacobian_rev = model(y_rev, rev=True)
            output_rev_rand, jacobian_rev_rand = model(y_rev_rand, rev=True)
    
            l_rev = (
                lambd_rev
                * loss_factor
                * loss_backward(output_rev_rand[:, :ndim_x],
                                x[:, :ndim_x], device=args.device)
            )
    
            l_rev += lambd_predict * loss_fit(output_rev, x)
            
            l_tot += l_rev.data.item()
            l_rev.backward()
    
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-15.00, 15.00)
    
            optimizer.step()
            
        cur_loss = l_tot / batch_idx
        t.set_description('training loss=%g' % (cur_loss))
        scheduler.step(cur_loss)
        
        if (i_epoch+1)%args.savemodel_period == 0 and i_epoch > 0:
            if cur_loss < loss_min:
                loss_min = cur_loss
                best_model_state = model.state_dict()
                print('Saving progress, saving the current best model, at iteration {}, best loss {}'.format(i_epoch, loss_min))
                
            save_name = '%s_noise_%.4e_checkpoint'%(dataset_name, noise_sigma)
            save_dict = {
                'save_path': args.path + save_name,
                'best_iteration': i_epoch,
                'best_loss': loss_min,
                'best_model_state_dict': best_model_state,
                'cur_model_state_dict': model.state_dict(),
            }
            utils.save_state(save_dict)
            # print & save loss & val_loss
            perf_eval = 'Iteration [{}/{}], Ly: {:.4f}, Lqp: {:.4f}, Lx: {:.4f}, cur_loss:{:.4f}, training_Loss_min: {:.4f}'.format(i_epoch, n_epochs, Ly.data.item(), Lpq.data.item(), l_rev.data.item(), cur_loss, loss_min)
            print(perf_eval, file=open(args.path + '%s_noise_%.4e_losses.txt'%(dataset_name, noise_sigma), "a"))
        

def test_INN_model(model, dataset_name,  noise_sigma = 0.0, plot=True, plot_circle=True):
    model.eval()

    dataset = data_LD.synth_dataset_LD(dataset_name)
    
    ndim_x = dataset.n
    ndim_y = dataset.m
    ndim_z = dataset.n - dataset.m
    
    zeros_noise_scale = 5e-2
    if dataset_name == 'concentric_spheres':
        # y_list = np.arange(0, 1.99, 0.02) 
        y_list = np.array([0.3, 0.9, 1.6])
    else:
        # y_list = np.arange(0, 1.01, 0.02) 
        y_list = np.array([0.3, 0.6, 0.95])
    fig, axes = plt.subplots(2, len(y_list), figsize=(4*len(y_list), 8))
    
    test_loaders = []
    xs = []
    ys = []
    
    Ly, Lpq = np.zeros(len(y_list)), np.zeros(len(y_list))
    
    display_nums = int(1e5)
    batch_size = 1000
    for i, cond_y in enumerate(y_list):
        x, y = dataset.generate_dataset(num_samples=display_nums, noise_sigma=noise_sigma, condition_y =  cond_y* np.ones(display_nums))
        test_loader = utils.get_dataloader(dataset,batch_size=batch_size, device=args.device)
        test_loaders.append(test_loader)
        xs.append(x)
        ys.append(y)
    
    
    for i in trange(len(test_loaders)):
        ly = []
        lpq = []
        for tl, (x, y) in enumerate(test_loaders[i]):
            y = y.reshape(-1, 1)
            y_samps = torch.cat([torch.randn(batch_size, ndim_z, device=args.device),
                                 zeros_noise_scale * torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z, device=args.device), y], dim=1)
        
            x_hat, rev_jac = model(y_samps, rev=True)
            pred_y = x_hat[:, :ndim_y]
            true_y = y_samps[:, -ndim_y:]
            ly.append(((pred_y - true_y)**2).squeeze().mean().cpu().data.numpy())
            lpq.append(inn.MMD_multiscale(x_hat[:,:ndim_x], x[:,:ndim_x], device=args.device).mean().cpu().data.numpy())
            
            if plot:
                x_hat = x_hat.cpu().data.numpy()
                heatmap_grid = 80
                title = '$p^*(x_1|y=%.2f)$'%(y_list[i]) if dataset_name == 'circle' else '$x\sim p^*(x|y=%.2f)$'%(y_list[i])
                axes[0, i].set_title(title, fontsize=20)
                title =  '$\widehat{p}(g(y=%.2f, z)_1)$'%(y_list[i]) if dataset_name == 'circle' else '$g(y=%.2f, z), z\sim N(0,I)$'%(y_list[i])
                axes[1, i].set_title(title, fontsize=20)
                
                if plot_circle and tl == 0:
                    dataset.display_dataset(axes[0, i], xs[i], ys[i], display_nums=len(ys[i]), color='r', heatmap=True, heatmap_grid=heatmap_grid, plot_circle=True)
                    dataset.display_dataset(axes[1, i], x_hat, ys[i], display_nums=len(ys[i]), heatmap=True, heatmap_grid=heatmap_grid, plot_circle=True)
                elif dataset_name != 'circle':
                    dataset.display_dataset(axes[0, i], xs[i], ys[i], display_nums=len(ys[i]), color='r', heatmap=True, heatmap_grid=heatmap_grid, plot_circle=False)
                    dataset.display_dataset(axes[1, i], x_hat, ys[i], display_nums=len(ys[i]), heatmap=True, heatmap_grid=heatmap_grid, plot_circle=False)
            
        Ly[i] = np.array(ly).mean()
        Lpq[i] = np.array(lpq).mean()
        
    if plot:
        fig.tight_layout()
        fig.savefig('%s%s_noise_%.4e_real_generated_samples.png'%(args.fig_path, args.dataset_name, args.noise_sigma))
    
    np.savez('%s%s_noise_%.4e_errors_temp'%(args.fig_path, args.dataset_name, args.noise_sigma),
             Ly = Ly, Lpq=Lpq, y_list=y_list)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="INN experiment")
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda:0','cuda:1'])
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--dataset_name', type=str, default='intersect_spheres', choices=[ 'intersect_spheres', 'circle', 'concentric_spheres', 'single_sphere'])
    parser.add_argument('--noise_sigma', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=int(1e5))
    parser.add_argument('--savemodel_period', type=int, default=10)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    
    args = parser.parse_args()
    
    if not os.path.exists('synthetic_dataset/INN'):
        os.makedirs('synthetic_dataset/INN')
    if not os.path.exists('synthetic_dataset/INN_figures'):
        os.makedirs('synthetic_dataset/INN_figures')
    args.path = 'synthetic_dataset/INN/'
    args.fig_path = 'synthetic_dataset/INN_figures/'
    
    dataset_name = args.dataset_name
    num_samples = args.num_samples
    noise_sigma = args.noise_sigma
    device = args.device
    
    batch_size = 1000
    ndim_tot = 16
    
    model = build_INN_model(ndim_tot)
    if args.train > 0:
        train_INN_model(model, dataset_name, num_samples=num_samples, noise_sigma=noise_sigma, batch_size=batch_size)
    if args.test > 0:
        _, _, model_state = utils.load_state(args.path + '%s_noise_%.4e_checkpoint.pt'%(dataset_name, noise_sigma), load_best=False) # For INN, need to load the last model.
        model.load_state_dict(model_state)
        test_INN_model(model, dataset_name, noise_sigma=noise_sigma)
    
    
    
    
    