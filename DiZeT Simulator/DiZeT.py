from itertools import combinations
from numpy.random import normal as randn
import numpy as np
import torch.nn as nn
import torch

class Encoder(torch.nn.Module):
    
    def __init__(self, K: int, device: str = "cpu"):
        super(Encoder, self).__init__()
        r = np.sqrt(1+np.sin(np.pi/K))
        self.shuffle_vector = (2*np.pi*torch.arange(0, K)/K).to(device = device)
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
    
    def __init__(self, K: int, return_bit_vector: bool = True, device: str = "cpu"):
        super(Decoder, self).__init__()
        r = np.sqrt(1+np.sin(np.pi/K))
        self.shuffle_vector = 2*np.pi*torch.arange(0, K)/K
        self.zero_bit_0 = 1/r*torch.exp(1j*self.shuffle_vector)
        self.zero_bit_1 = r*torch.exp(1j*self.shuffle_vector)
        self.K = K
        self.R = r
        self.curr_zeros = None
        self.return_bit_vector = return_bit_vector
        self.device = device
    
    def get_index(self, m):
        base_10 = m * 2**torch.arange(self.K-1, -1, -1)
        return torch.sum(base_10, dim = -1)

    def horner(self, eval_point, x):
        result = x[:, 0]
        for i in range(1, self.K+1):
            result = result*eval_point + x[:, i]
        return torch.abs(result)

    def decode(self, x):
        b, _ = x.shape
        m = torch.zeros((b, self.K))
        for i in range(self.K):
            m[:,i] =  torch.where(self.R**self.K * self.horner(self.zero_bit_0[i], x) >= self.horner(self.zero_bit_1[i], x), 1, 0)
        return m

    def forward(self, x: torch.Tensor):
        m = self.decode(x)
        return (m if self.return_bit_vector else self.get_index(m))

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
    
def test():
    dec = Decoder(4)
    poly = torch.Tensor([[1,1,1,1,1]])
    val = 1j+2

    print(dec.horner(val, poly))
    return

if __name__ == "__main__":
    test()