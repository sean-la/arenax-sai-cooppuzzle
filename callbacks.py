import logging
import numpy as np
import torch as torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym



class PretrainingCallback(BaseCallback):
    """Callback for behavioral cloning pre-training phase"""
    def __init__(self, demonstrations, n_epochs=10, batch_size=64, lr=1e-4,
                 pretrain_save_location=None):
        super().__init__()
        self.demonstrations = demonstrations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pretrain_completed = False
        self.pretrain_save_location = pretrain_save_location
    
    def _on_training_start(self):
        if not self.pretrain_completed:
            logging.info("Starting behavioral cloning pre-training...")
            self._pretrain_behavioral_cloning()
            self.pretrain_completed = True
            if self.pretrain_save_location is not None:
                logging.info(f"Saving pretrained model to path {self.pretrain_save_location}")
                self.model.save(self.pretrain_save_location)
    
    def _pretrain_behavioral_cloning(self):
        # Convert demonstrations to tensors
        demo_obs = torch.FloatTensor(np.array([d['observation'] for d in self.demonstrations]))
        demo_actions = torch.FloatTensor(np.array([d['action'] for d in self.demonstrations]))
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(demo_obs, demo_actions)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Get policy network
        #policy_net = self.model.policy.action_net
        #optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        #optimizer = self.model.policy.optimizer
        
        # Training loop
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                self.model.policy.optimizer.zero_grad()
                
                ## Get policy distribution
                #features = self.model.policy.extract_features(batch_obs)
                #latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                #distribution = self.model.policy.action_dist.proba_distribution(latent_pi)
                
                ## Compute loss (negative log likelihood)
                #loss = -distribution.log_prob(batch_actions).mean()
                pi_demo = self.model.policy.get_distribution(demo_obs)
            
                # Compute demonstration loss (behavior cloning loss)
                loss = -pi_demo.log_prob(demo_actions).mean()
                
                # Update policy
                loss.backward()
                self.model.policy.optimizer.step()
                
                total_loss += loss.item()
            
            logging.info(f"Pretraining Epoch {epoch+1}/{self.n_epochs}, "
                  f"Loss: {total_loss/len(dataloader):.4f}")
            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    logging.debug(f"Layer: {name} | Gradient: \n{param.grad}")
    
    def _on_step(self):
        return True



class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        Return False to stop training.
        """
        self.env.render()
        return True

