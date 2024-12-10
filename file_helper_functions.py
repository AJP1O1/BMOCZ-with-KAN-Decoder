from scipy.io import savemat
from tkinter import filedialog
import tkinter as tk
import numpy as np
import torch.nn as nn
import torch
import os

def save_to_matlab(bler: list, file_name: str = "BLER_kan", os_path: str = None):
    if(os_path is not None):
        save_path = os_path
    else:
        save_path = os.path.dirname(os.path.realpath(__file__))
    dBlist = np.array(list(bler.keys()))
    BLERs = np.array(list(bler.values()))
    matlab_file = np.vstack((dBlist, BLERs))
    savemat(save_path + f"\\{file_name}.mat", {f'{file_name}_matrix': matlab_file})

def load_model(os_path: str = None, auto_load: bool = False):
    if os_path is None and auto_load is True:
        load_path = os.path.dirname(os.path.realpath(__file__))
    elif os_path is not None and auto_load is True:
        load_path = os_path
    else:
        tk.Tk().withdraw()
        load_path = filedialog.askopenfilename(
            initialdir = os.path.dirname(os.path.realpath(__file__)),
            title = "Select a File",
            filetypes = ((".pth files", "*.pth*"),("All files", "*.*")))
    return torch.load(load_path)

def get_learnable_parameters(model: nn.Module):
    learnable = sum([p.numel() for p in model.parameters()])
    print(f"There are {learnable} learnable parameters in this model")

def get_shuffle_vector(encoder: nn.Module):
    return encoder.shuffle_vector.data

def get_radius(encoder: nn.Module):
    return encoder.R.data