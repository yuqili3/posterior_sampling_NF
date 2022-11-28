# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:16:09 2022

@author: liyuq
"""
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch



class synth_dataset_LD:
    def __init__(self, dataset_name, number_of_spheres=2):
        
        '''
        datasets:
            1. circle: 2D->1D
            2. single_sphere: 3D->1D
            3. concentric spheres: 3D->1D, disconnected support for signal, disconnected for S(y)
            4. intesect_spheres: 3D->1D, connected support for signal, disconnected for S(y)
        '''
        self.dataset_name = dataset_name
        if self.dataset_name == 'circle':
            self.n = 2
            self.m = 1
            self.signal_connected=True
            self.posterior_connected =False
        elif self.dataset_name == 'single_sphere':
            self.n = 3
            self.m = 1
            self.signal_connected=True
            self.posterior_connected = True
            self.number_of_spheres = 1
        elif self.dataset_name == 'concentric_spheres':
            self.n = 3
            self.m = 1
            self.signal_connected=False
            self.posterior_connected =False
            self.number_of_spheres = number_of_spheres
        elif self.dataset_name == 'intersect_spheres':
            self.n = 3
            self.m = 1
            self.signal_connected=True
            self.posterior_connected =False
            self.number_of_spheres = 2
            
    def generate_dataset(self, num_samples=1000, noise_sigma=0.0, condition_y=None):
        '''
        num_samples: only used when condition_y is not provided
        condition_y: is a np array containing the measurements
        If the measurement y is provided, generate x corresponding to the same condition_y
        Else generate unconditional distribution
        '''
        self.noise_sigma = noise_sigma
        
        if self.dataset_name == 'circle':
            list_condition = [-1,1]
            list_condition_prob = [1/2, 1/2]
            
            if condition_y is not None:
                idx_condition = np.random.choice(list_condition, len(condition_y), p=list_condition_prob)
                x_coord = condition_y - self.noise_sigma*np.random.randn(*condition_y.shape)
                x_coord = np.clip(x_coord, -1, 1)
                x_radius = np.sqrt(1 - x_coord**2)
                x_point = np.c_[condition_y, idx_condition * x_radius]
                x = np.array(x_point, dtype='float32')
                y = np.array(condition_y, dtype='float32')
                
            else:
                unit_vec = np.random.randn(num_samples, self.n)
                unit_vec = unit_vec / np.linalg.norm(unit_vec, axis=1, keepdims=True)
                y = unit_vec[:,0] + self.noise_sigma*np.random.randn(num_samples)
                x = unit_vec
            
        
        elif self.dataset_name == 'single_sphere':
            
            if condition_y is not None:
                angle = 2 *np.pi *np.random.rand(len(condition_y))
                x_coord = condition_y - self.noise_sigma*np.random.randn(*condition_y.shape)
                x_coord = np.clip(x_coord, -1, 1)
                x_radius = np.sqrt(1 - x_coord**2)
                x_point = np.c_[condition_y, x_radius * np.cos(angle), x_radius * np.sin(angle)]
                x = np.array(x_point, dtype='float32')
                y = np.array(condition_y, dtype='float32')
                
            else:
                unit_vec = np.random.randn(num_samples, self.n)
                unit_vec = unit_vec / np.linalg.norm(unit_vec, axis=1, keepdims=True)
                x = unit_vec
                y = unit_vec[:,0] + self.noise_sigma*np.random.randn(num_samples)
                
            
            
        elif self.dataset_name == 'concentric_spheres':
            r_list = np.arange(1, self.number_of_spheres+1)
            
            if condition_y is not None:
                angle = 2 *np.pi *np.random.rand(len(condition_y))
                x_coord = condition_y - self.noise_sigma*np.random.randn(*condition_y.shape)
                x_coord = np.clip(x_coord, -self.number_of_spheres, self.number_of_spheres)
                point_projections = np.zeros((len(condition_y), self.n - 1))
                for xi, x_c in enumerate(x_coord):
                    radii_list = np.sqrt(r_list ** 2 - x_c**2) # will contain nan value
                    radii_probs = np.nan_to_num(radii_list / r_list ** 2 , nan=0)
                    radii_probs /= np.sum(radii_probs)
                    radius = np.random.choice(radii_list, size=1, p = radii_probs)
                    unit_vec = np.random.randn(self.n - 1)
                    point_projections[xi] = radius * unit_vec / np.linalg.norm(unit_vec)
                x = np.c_[condition_y, point_projections]
                y = condition_y
                
            else:
                r = np.random.choice(r_list, num_samples, p=[1/self.number_of_spheres]*self.number_of_spheres)
                ### condition here is referring to the first coordinate of the sphere
                unit_vec = np.random.randn(num_samples, self.n)
                point_3d = (r * unit_vec.T / np.linalg.norm(unit_vec, axis=1)).T# 3d point on the sphere
                x = point_3d
                y = point_3d[:, 0] + self.noise_sigma * np.random.randn(num_samples)
                
        elif self.dataset_name == 'intersect_spheres':
            radius = 1
            center_list = np.array([[0, -0.75, 0], [0, 0.75, 0]])
            n_centers = len(center_list)
            if condition_y is not None:
                angle = 2 *np.pi *np.random.rand(len(condition_y))
                x_coord = condition_y - self.noise_sigma*np.random.randn(*condition_y.shape)
                x_coord = np.clip(x_coord, -1, 1)
                x_radius = np.sqrt(radius ** 2 - x_coord**2)
                
                unit_vec = np.random.randn(len(condition_y), self.n-1)
                idx_condition = np.random.choice(np.arange(n_centers), len(condition_y), p=[1/n_centers]*n_centers)
                point_proj = (x_radius * unit_vec.T / np.linalg.norm(unit_vec, axis=1)).T + center_list[idx_condition][:,1:]
                x = np.c_[condition_y, point_proj]
                y = condition_y
                
            else:
                idx_condition = np.random.choice(np.arange(n_centers), num_samples, p=[1/n_centers]*n_centers)
                ### condition here is referring to the first coordinate of the sphere
                unit_vec = np.random.randn(num_samples, self.n)
                point_3d = (radius * unit_vec.T / np.linalg.norm(unit_vec, axis=1)).T 
                point_3d += center_list[idx_condition]
                x = point_3d
                y = point_3d[:, 0] + self.noise_sigma * np.random.randn(num_samples)
                
            
                
                
        self.x, self.y = x, y
        return x, y
        
            
        
        
        
    def display_dataset(self, ax, x, y, display_nums=20, spheres_radii=None, color=None,linewidth=1, heatmap=True,heatmap_grid=70, plot_circle=True):
        '''
        assumes x, y is numpy array 
        '''
        if self.dataset_name == 'circle':
            lim_x_l, lim_y_l, lim_x_u, lim_y_u = -1.5, -1.5, 1.5, 1.5
            ax.grid('on')
            
            
            if heatmap:
                heatmap, xedges = np.histogram(x[:, 1], 
                    range=[lim_x_l, lim_x_u],
                    bins=np.linspace(lim_x_l, lim_x_u, heatmap_grid),
                    density=True)
                ax.plot((xedges[:-1] + xedges[1:])/2, heatmap,'.--', linewidth=1)
                # ax.set_xlabel('$x_1$')
                # ax.set_ylabel('$p^*(x|y)$')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.set_xlim([lim_x_l,lim_x_u])
                ax.set_ylim([lim_y_l,lim_y_u])
                if plot_circle:
                    # plot a unit circle
                    degrees = np.arange(0, 2*np.pi, 0.01) 
                    xc, yc = np.cos(degrees), np.sin(degrees)
                    radii_list = [1]
                    for r in radii_list:
                        ax.plot(r*xc, r*yc, '-k', linewidth=1, alpha=0.5)
                ax.set_xlabel('y')
                alpha = 0.8 if color == 'r' else 0.5
                for t in range(display_nums):
                    assigned_color = 'C%d'%(np.random.randint(10)) if not color else color
                    ax.plot(x[t,0], x[t,1], 'o%s'%(assigned_color), alpha=alpha)
                    ax.plot(y[t], 0, '.%s'%(assigned_color), linewidth, alpha=alpha)
                    ax.plot([x[t,0], y[t]], [x[t,1], 0], ':%s'%(assigned_color), linewidth=linewidth, alpha=alpha)
            
            
        elif self.dataset_name == 'single_sphere':
            ## ASSUMES condition_y is taking a single value
            condition_y = np.mean(y)
            lim_x_l, lim_y_l, lim_x_u, lim_y_u = -1.5, -1.5, 1.5, 1.5
            ax.grid('on')
            ax.set_xlim([lim_x_l,lim_x_u])
            ax.set_ylim([lim_y_l,lim_y_u])
            if plot_circle:
                # plot a unit circle
                degrees = np.arange(0, 2*np.pi, 0.01) 
                xc, yc = np.cos(degrees), np.sin(degrees)
                radii_list = [1]
                for r in radii_list:
                    if condition_y < r:
                        x_radius = np.sqrt(r**2 - condition_y**2)
                        ax.plot(x_radius*xc, x_radius*yc, '-k', linewidth=1, alpha=0.5)  
            # ax.set_xlabel('$x_1$')
            # ax.set_ylabel('$x_2$')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            if heatmap:
                heatmap, xedges, yedges = np.histogram2d(
                    x[:, 1], x[:, 2], 
                    range=[[lim_x_l, lim_x_u],[lim_y_l, lim_y_u ]],
                    bins=(np.linspace(lim_x_l, lim_x_u, heatmap_grid), np.linspace(lim_y_l, lim_y_u,heatmap_grid)))
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds')
            else:
                alpha = 0.5 if color == 'r' else 0.3
                for t in range(display_nums):
                    assigned_color = 'C%d'%(np.random.randint(10)) if not color else color
                    ax.plot(x[t,1], x[t,2], 'o%s'%(assigned_color), alpha=alpha)
            
            
        elif self.dataset_name == 'concentric_spheres':
            condition_y = np.mean(y)
            lim_l, lim_u= -self.number_of_spheres-0.5, self.number_of_spheres+0.5
            ax.grid('on')
            ax.set_xlim([lim_l, lim_u])
            ax.set_ylim([lim_l, lim_u])
            
            if plot_circle:
                degrees = np.arange(0, 2*np.pi, 0.01) 
                xc, yc = np.cos(degrees), np.sin(degrees)
                radii_list = np.arange(1, self.number_of_spheres+1)
                for r in radii_list:
                    if condition_y < r:
                        x_radius = np.sqrt(r**2 - condition_y**2)
                        ax.plot(x_radius*xc, x_radius*yc, '-k', linewidth=1, alpha=0.5)  
            # ax.set_xlabel('$x_1$')
            # ax.set_ylabel('$x_2$')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            if heatmap:
                heatmap, xedges, yedges = np.histogram2d(
                    x[:, 1], x[:, 2], 
                    range=[[lim_l, lim_u],[lim_l, lim_u]],
                    bins=(np.linspace(lim_l, lim_u, heatmap_grid), np.linspace(lim_l, lim_u,heatmap_grid)))
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds')
            else:
                alpha = 0.5 if color == 'r' else 0.3
                for t in range(display_nums):
                    assigned_color = 'C%d'%(np.random.randint(10)) if not color else color
                    ax.plot(x[t,1], x[t,2], 'o%s'%(assigned_color), alpha=alpha)

        
        elif self.dataset_name == 'intersect_spheres':
            condition_y = np.mean(y)
            ax.grid('on')
            lim_x_l, lim_y_l, lim_x_u, lim_y_u = -2, -1.5, 2, 1.5
            ax.set_xlim([-2, 2])
            ax.set_ylim([-1.5, 1.5])
            
            if plot_circle:
                degrees = np.arange(0, 2*np.pi, 0.01) 
                xc, yc = np.cos(degrees), np.sin(degrees)
                x_radius = np.sqrt(1 - condition_y**2)
                ax.plot(x_radius*xc + 0.75, x_radius*yc, '-k', linewidth=1, alpha=0.5)  
                ax.plot(x_radius*xc - 0.75, x_radius*yc, '-k', linewidth=1, alpha=0.5)  
            # ax.set_xlabel('$x_1$')
            # ax.set_ylabel('$x_2$')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            if heatmap:
                heatmap, xedges, yedges = np.histogram2d(
                    x[:, 1], x[:, 2], 
                    range=[[lim_x_l, lim_x_u],[lim_y_l, lim_y_u ]],
                    bins=(np.linspace(lim_x_l, lim_x_u, heatmap_grid), np.linspace(lim_y_l, lim_y_u,heatmap_grid)))
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='Reds')
            else:
                alpha = 0.5 if color == 'r' else 0.3
                for t in range(display_nums):
                    assigned_color = 'C%d'%(np.random.randint(10)) if not color else color
                    ax.plot(x[t,1], x[t,2], 'o%s'%(assigned_color), alpha=alpha)
            
        
    
if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    
    dataset = synth_dataset_LD('circle')
    # x, y = dataset.generate_dataset(1000, noise_sigma=0.00)
    x, y = dataset.generate_dataset(noise_sigma=0.01, condition_y = 0.7 * np.ones(100))
    dataset.display_dataset(ax,x, y, display_nums=50)
    
    # dataset = synth_dataset_LD('single_sphere')
    # x, y = dataset.generate_dataset(noise_sigma=0.00, condition_y = 0.7 * np.ones(5000))
    # dataset.display_dataset(ax, x, y, display_nums=1000, spheres_radii=[1])
    
    # dataset = synth_dataset_LD('concentric_spheres')
    # # x, y = dataset.generate_dataset(noise_sigma=0.02)
    # x, y = dataset.generate_dataset(noise_sigma=0.02, condition_y = 0.1 * np.ones(5000))
    # dataset.display_dataset(ax, x, y, display_nums=100, spheres_radii=[1], heatmap=True)
    
    # dataset = synth_dataset_LD('intersect_spheres')
    # # x, y = dataset.generate_dataset(noise_sigma=0.02)
    # x, y = dataset.generate_dataset(noise_sigma=0.02, condition_y = 0.1 * np.ones(5000))
    # dataset.display_dataset(ax, x, y, display_nums=100, spheres_radii=[1])
    
    