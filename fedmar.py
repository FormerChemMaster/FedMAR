import torch
import numpy as np

class FedMAR:
    def __init__(self, global_model, p=2, alpha_dir=0.5):
        self.global_model = global_model
        self.p = p  
        self.alpha_dir = alpha_dir  
        self.S_ch = []    
        self.S_p = []     
        self.S_fr = []    
        self.weights = {} 
        
    def async_aggregation(self, updates, data_sizes):

        data_sizes = np.array(data_sizes)
        weights = np.exp(data_sizes) / np.sum(np.exp(data_sizes))
        
        aggregated = None
        for i, (client_id, update) in enumerate(updates.items()):
            if i == 0:
                aggregated = torch.zeros_like(update)
            aggregated = (aggregated * sum(weights[:i]) + weights[i]*update) / sum(weights[:i+1])
        return aggregated
    
    def detect_malicious(self, updates, sigma_dp):
        updates = list(updates.values())
        centroid = torch.mean(torch.stack(updates), dim=0)
        deviations = [torch.norm(u - centroid, p=self.p) for u in updates]
        sigma = np.std(deviations)
        for client_id, update in updates.items():
            dev = torch.norm(update - centroid, p=self.p)
            if dev > 3*sigma:
                self.S_p.append(client_id)
            elif dev < sigma_dp:
                self.S_fr.append(client_id)
            else:
                self.S_ch.append(client_id)
    def rollback(self, aggregated, updates, client_id):
        w_total = sum(self.weights.values())
        w_client = self.weights[client_id]
        return (aggregated * w_total - w_client * updates[client_id]) / (w_total - w_client)
    
    def fine_tuning(self, global_update, prev_update, feature_extractor):
        param_diff = torch.norm(global_update - prev_update, p=2)
        feature_diff = torch.norm(feature_extractor(global_update) - feature_extractor(prev_update), p=2)
        
        alpha = 1 / max(1, feature_diff.item())
        beta = 1 / max(1, param_diff.item())
        return alpha * feature_diff + beta * param_diff