import torch
import numpy as np

class ServerPrivacy:
    def __init__(self, epsilon, alpha_r=2):
        self.epsilon = epsilon
        self.alpha_r = alpha_r
    
    def calculate_sensitivity(self, model):
        sensitivities = []
        for param in model.parameters():
            sensitivities.append(torch.max(torch.abs(param.data)))
        return max(sensitives)
    
    def add_noise(self, aggregated, sensitivity):
        sigma_gaussian = np.sqrt(self.alpha_r * sensitivity**2 / (2 * self.epsilon))
        lambda_laplace = (self.alpha_r - 1) * sensitivity / self.epsilon

        if sigma_gaussian < lambda_laplace/np.sqrt(2):
            noise = torch.normal(0, sigma_gaussian, aggregated.shape)
        else:
            noise = torch.distributions.Laplace(0, lambda_laplace).sample(aggregated.shape)
        
        return aggregated + noise
    
    def clip_gradients(self, gradients, clip_bound):
        norm = torch.norm(gradients)
        if norm > clip_bound:
            return gradients * clip_bound / norm
        return gradients