# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:25:41 2022

@author: liyuq
"""

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    methods = ['INN', 'CAF']
    paths = ['synthetic_dataset/INN_figures/', 'synthetic_dataset/HCNAF_figures/']
    datasets = ['intersect_spheres', 'circle', 'concentric_spheres', 'single_sphere']
    noise_sigmas = [0]
    
    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(8,4))

        for method, path in zip(methods, paths):
            for noise_sigma in noise_sigmas:
                npzfile = np.load('%s%s_noise_%.4e_errors.npz'%(path, dataset, noise_sigma))
                Ly= npzfile['Ly']
                Lpq=npzfile['Lpq']
                y_list=npzfile['y_list']
                
                # print(method, Ly)
                
                
                axes[0].plot(y_list, Ly,'x:', label=method)
                
                axes[1].plot(y_list, Lpq,'o:', label=method)
                
                axes[0].set_title('$||y - A(g(y, z))||_2^2$', fontsize=20)
                axes[0].set_xlim([np.min(y_list)-0.1,np.max(y_list)+0.1])
                axes[0].set_yscale('log')
                axes[0].set_xlabel('y', fontsize=20)
                axes[0].grid('on')
                
                axes[1].set_title('$D(p^*(x|y), \widehat{p}(x|y))$', fontsize=20)
                axes[1].set_xlim([np.min(y_list)-0.1,np.max(y_list)+0.1])
                axes[1].set_yscale('log')
                axes[1].grid('on')
                axes[1].set_xlabel('y', fontsize=20)
                
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig('%s_noise_%.4e_compare_metrics.png'%(dataset, noise_sigma))
    