import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import unet_segdiff 

class DensityFlowMatching(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        
        self.vector_field_net = unet_segdiff.UNetModel3D(
            in_channels=2,
            model_channels=64,
            out_channels=1,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            dims=3,
            use_checkpoint=False,
            num_heads=1,
            use_scale_shift_norm=True,
        )
        self.dropout_rate = 0.1
        self.feature_dropout_rate = 0.05
        self.output_dropout_rate = 0.05

        self.dropout = nn.Dropout3d(self.dropout_rate)
        self.feature_dropout = nn.Dropout3d(self.feature_dropout_rate)
        self.output_dropout = nn.Dropout3d(self.output_dropout_rate)
        
        self.initial_mean = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        self.initial_std = nn.Parameter(torch.ones(1, 1, 1, 1, 1))

        self.t_eps=0.001

    def get_conditional_initial_distribution(self, density_condition):
        
        noise_tensor = torch.randn(density_condition.shape, device=density_condition.device)
        return noise_tensor

    def get_conditional_interpolation(self, density_condition, clean_prob_map, t):
        
        batch_size = density_condition.shape[0]
        t = t.view(-1, 1, 1, 1, 1)
        
        initial_distribution = self.get_conditional_initial_distribution(density_condition)
        
        prob_t = (1 - t) * initial_distribution + t * clean_prob_map
        
        return prob_t, initial_distribution

    def forward(self, density_condition, prob_t, t):
        
        network_input = torch.cat([density_condition, prob_t], dim=1)
        if self.training and self.dropout_rate > 0:
            network_input = self.dropout(network_input)
        
        pred_vector_field = self.vector_field_net(network_input, t)

        if self.training and self.output_dropout_rate > 0:
            pred_vector_field = self.output_dropout(pred_vector_field)
        
        return pred_vector_field

    @torch.no_grad()
    def sample(self, density_condition, num_steps=10, method='euler', return_all_steps=False):
        
        all_outputs = []
        batch_size = density_condition.shape[0]
        device = density_condition.device
        
        prob_t = self.get_conditional_initial_distribution(density_condition)
        if return_all_steps:
            all_outputs.append(prob_t.detach().clone())
        
        
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        
        trajectory = [prob_t.detach().cpu()]
        trajectory_times = [0.0]

        if return_all_steps:
            all_outputs.append(prob_t.detach().clone())
        
        for i in range(num_steps):
            t_current = timesteps[i]
            
            pred_vector_field = self(
                density_condition, 
                prob_t, 
                torch.tensor([t_current] * batch_size, device=device)
            )
            
            if method == 'euler':
                prob_t = prob_t + pred_vector_field * dt
            elif method == 'heun':
                
                prob_t_pred = prob_t + pred_vector_field * dt
                
                pred_vector_field_corr = self(
                    density_condition, 
                    prob_t_pred, 
                    torch.tensor([t_current + dt] * batch_size, device=device)
                )
                prob_t = prob_t + 0.5 * (pred_vector_field + pred_vector_field_corr) * dt
            else:
                raise ValueError(f"Unknown ODE solver: {method}")
            
            
            if i % 20 == 0 or i == num_steps - 1:
                trajectory.append(prob_t.detach().cpu())
                trajectory_times.append(t_current.item() + dt)

            if return_all_steps:
               
                all_outputs.append(pred_vector_field.detach().clone())
                all_outputs.append(prob_t.detach().clone())
        
        final_prob_map = torch.clamp(prob_t, 0, 1)
        
        if return_all_steps:
            return final_prob_map, all_outputs, trajectory, trajectory_times
        else:
            return final_prob_map, trajectory, trajectory_times

    @torch.no_grad()
    def sample_with_uncertainty(self, density_condition, num_samples=3, num_steps=100):
        
        all_samples = []
        
        for i in range(num_samples):
            if self.logger:
                self.logger.info(f"Uncertainty sampling - Sample {i+1}/{num_samples}")
            
            
            prob_t = self.get_conditional_initial_distribution(density_condition)
            
            dt = 1.0 / num_steps
            timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=density_condition.device)
            
            for step in range(num_steps):
                t_current = timesteps[step]
                pred_vector_field = self.vector_field_net(
                    torch.cat([density_condition, prob_t], dim=1),
                    torch.tensor([t_current] * density_condition.shape[0], device=density_condition.device)
                )
                prob_t = prob_t + pred_vector_field * dt
                prob_t = torch.clamp(prob_t, 0, 1)
            
            all_samples.append(prob_t.detach().cpu())
        
       
        samples_tensor = torch.stack(all_samples)  # [num_samples, B, 1, D, H, W]
        mean_pred = samples_tensor.mean(dim=0)
        std_pred = samples_tensor.std(dim=0)
        
        return mean_pred, std_pred, all_samples
