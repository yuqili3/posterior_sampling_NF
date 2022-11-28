# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:43:08 2022

@author: liyuq
"""


import torch
import torch.nn as nn
import torch.optim

def save_state(save_dict):
    print('Saving model_state_dict...')
    print('best iteration: {}, loss: {}'.format(save_dict['best_iteration'], save_dict['best_loss']))
    torch.save({
            'best_iteration': save_dict['best_iteration'],
            'best_loss': save_dict['best_loss'],
            'best_model_state_dict': save_dict['best_model_state_dict'],
            'cur_model_state_dict': save_dict['cur_model_state_dict']
        }, save_dict['save_path'] + '.pt')     


def load_state(path, load_best = False):
    print('Loading model_state_dict, optimizer_state_dict, scheduler_state_dict..')
    save_dict = torch.load(path)
    best_iteration = save_dict['best_iteration']
    best_loss = save_dict['best_loss']
    if load_best:
        model_state = save_dict['best_model_state_dict']
    else:
        model_state = save_dict['cur_model_state_dict']

    return best_iteration, best_loss, model_state


def get_dataloader(dataset, batch_size = 1000, shuffle=True, device='cuda:0'):
    toTensor = lambda x: torch.from_numpy(x).float().to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(toTensor(dataset.x), toTensor(dataset.y)),
        batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader
