from kan import KAN
from itertools import combinations
from numpy.random import normal as randn
import numpy as np
import torch.nn as nn
import torch

class Encoder(torch.nn.Module):
    
    def __init__(self, K: int, learnable_radius: bool = False, device: str = "cpu"):
        super(Encoder, self).__init__()
        if learnable_radius:
            r = nn.Parameter(torch.Tensor(1))
            nn.init.uniform_(r)
        else:
            r = np.sqrt(1+np.sin(np.pi/K))
        self.shuffle_vector = nn.Parameter(torch.Tensor(K))
        nn.init.uniform_(self.shuffle_vector)
        self.K = K
        self.R = r
        self.messages = self.get_messages().to(device = device)
        self.curr_zeros = None
        self.device = device

    def get_messages(self):
        messages = [format(x, f'0{self.K}b') for x in range(2**(self.K))]
        for i, message in enumerate(messages):
            messages[i] = [int(x) for x in list(message)]
        return torch.Tensor(messages).to(dtype = torch.float32)

    def compute_polynomial_coefficients(self, roots: torch.Tensor):
        b, k = roots.shape
        coefficients_batch = torch.zeros(b, k + 1, dtype = torch.complex128).to(device = self.device)
        coefficients_batch[:, 0] = 1
        for degree in range(1, k + 1):
            combs = combinations(range(k), degree)
            sign = (-1)**degree
            for comb in combs:
                product = torch.prod(roots[:, list(comb)], dim=1).to(device = self.device)
                coefficients_batch[:, degree] += sign * product
        return coefficients_batch
    
    def save_zeros(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        x = torch.flatten(x, start_dim = 1)
        self.curr_zeros = x
    
    def normalize_polynomial_coefficients(self, x: torch.Tensor):
        l2norm = torch.norm(x, p = 2, dim = -1)
        return x/l2norm.unsqueeze(-1) * np.sqrt(self.K + 1)

    def forward(self, x: torch.Tensor):
        radii = torch.where(x > 0, self.R, 1/self.R)
        zeros = radii.to(dtype = torch.complex128) * torch.exp(1j*self.shuffle_vector)
        self.save_zeros(zeros)
        coeffs = self.compute_polynomial_coefficients(zeros)
        return self.normalize_polynomial_coefficients(coeffs)

class Decoder(torch.nn.Module):
    
    def __init__(self, K: int, hidden_layers: int, device: str = "cpu"):
        super(Decoder, self).__init__()
        self.H = hidden_layers
        self.K = K
        self.Dec = KAN(
            layers_hidden=[2*K, hidden_layers, int(2**K)],  # Define input, hidden, and output sizes
            grid_size=5,  # Default grid size, adjust if necessary
            spline_order=3,  # Default spline order
            scale_noise=0.1,  # Default scale for noise
            scale_base=1.0,  # Default scale for base functions
            scale_spline=1.0,  # Default scale for spline functions
            base_activation=torch.nn.SiLU,  # Activation function
            grid_eps=0.02,  # Default epsilon for grid
            grid_range=[-1, 1],  # Default range for grid
        )
        self.C = self.companion_matrix()
        self.curr_zeros = None
        self.device = device
        self.norm = torch.nn.LayerNorm(2*K)

    def companion_matrix(self):
        n = self.K
        Cmat = torch.zeros((n, n), dtype = torch.complex128)
        Cmat[0, :] = 0
        Cmat[1:, :-1] = torch.eye(n-1, n-1, dtype = torch.complex128)
        return Cmat
    
    def save_zeros(self, x: torch.Tensor):
        x = torch.view_as_real(x)
        x = torch.flatten(x, start_dim = 1)
        self.curr_zeros = x

    def roots(self, coeffs: torch.Tensor):
        b, _ = coeffs.shape
        norm_coeffs = coeffs/(coeffs[:,0].unsqueeze(-1))
        c_coeffs = torch.flip(-1*norm_coeffs[:,1:], (-1,))
        C = (self.C.expand(b,-1,-1)).clone()
        C[:, :, -1] = c_coeffs
        eigs = torch.linalg.eigvals(C)
        roots = torch.view_as_real(eigs)
        return torch.flatten(roots, start_dim = 1)

    def forward(self, x: torch.Tensor):
        self.save_zeros(x)  
        roots = self.roots(x).to(dtype = torch.float32, device = self.device)
        return self.Dec(roots)

class Channel(torch.nn.Module):

    def __init__(self, K: int, EsNo: float, fading_channel: bool = False, device: str = "cpu"):
        super(Channel, self).__init__()
        self.K = K
        self.No = None
        self.sd = None
        self.update_snr(EsNo)
        self.channel = fading_channel
        self.device = device

    def update_snr(self, EsNo):
        self.No = 10**(-EsNo/10)
        self.sd = np.sqrt(self.No/2)

    def get_awgn_noise(self, b):
        return torch.from_numpy(randn(0.0, self.sd, size = (b, self.K + 1)) + 1j*randn(0.0, self.sd, size = (b, self.K + 1))).to(device = self.device)

    def get_channel_coefficient(self, b):
        return 1/np.sqrt(2)*torch.from_numpy(randn(0, 1, size = (b, 1)) + 1j*randn(0, 1, size = (b, 1))).to(device = self.device)

    def forward(self, x: torch.Tensor):
        b, _ = x.shape
        w = self.get_awgn_noise(b)
        h = self.get_channel_coefficient(b) if self.channel else 1.0
        return h*x + w