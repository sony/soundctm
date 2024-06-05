"""preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".
class VPPrecond:
    def __init__(self,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
        epsilon_t: float = 1e-5,
    ):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))


    def network_precond(self, sigma):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)
        
        return c_skip, c_out, c_in, c_noise

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def set_noise(self, latents: torch.FloatTensor, validation_mode: bool = False):
        if validation_mode:
            rng_state = torch.get_rng_state()
            torch.manual_seed(1234)

        rnd_uniform = torch.rand([latents.shape[0], 1, 1, 1], device=latents.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        loss_weight = 1 / sigma ** 2

        if validation_mode:
            torch.set_rng_state(rng_state)
        # n = torch.randn_like(latents) 
        return loss_weight, sigma

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VEPrecond:
    def __init__(self,         
        sigma_min: float = 0.02,        
        sigma_max: float = 100,         
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def network_precond(self, sigma):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()
        
        return c_skip, c_out, c_in, c_noise

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def set_noise(self, latents: torch.FloatTensor, validation_mode: bool = False):
        if validation_mode:
            rng_state = torch.get_rng_state()
            torch.manual_seed(1234)
            
        rnd_uniform = torch.rand([latents.shape[0], 1, 1, 1], device=latents.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        loss_weight = 1 / sigma ** 2
        
        if validation_mode:
            torch.set_rng_state(rng_state)

        return loss_weight, sigma

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

class iDDPMPrecond:
    def __init__(self,           
        C_1: float = 0.001,           
        C_2: float = 0.008,          
        M: int = 1000,
    ):
        super().__init__()
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        # self.register_buffer('u', u)
        self.u = u
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def network_precond(self, sigma):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        return c_skip, c_out, c_in, c_noise

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

    def set_noise(self, latents: torch.FloatTensor, validation_mode: bool = False):
        if validation_mode:
            rng_state = torch.get_rng_state()
            torch.manual_seed(1234)
            
        index = torch.randint(low=0, high=self.M - 1, size=(latents.shape[0], 1, 1, 1), device=latents.device)
        u = self.u.to(latents.device)
        sigma = u[index]
        loss_weight = 1 / sigma ** 2
        
        if validation_mode:
            torch.set_rng_state(rng_state)

        return loss_weight, sigma

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond:
    def __init__(self,
        sigma_min: float = 0.002,               
        sigma_max: float = 80.0,    
        sigma_data: float = 0.5,
        rho: float = 7.0, 
        P_mean: float = -1.2,              
        P_std:  float = 1.2,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.rho = rho

    def network_precond(self, sigma):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        return c_skip, c_out, c_in, c_noise

    def set_noise(self, latents: torch.FloatTensor, validation_mode: bool = False):
        if validation_mode:
            rng_state = torch.get_rng_state()
            torch.manual_seed(1234)
            
        rnd_normal = torch.randn([latents.shape[0], 1, 1, 1], device=latents.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        loss_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        if validation_mode:
            torch.set_rng_state(rng_state)
        
        return loss_weight, sigma

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
