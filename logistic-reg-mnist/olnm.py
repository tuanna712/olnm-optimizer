import math
import torch
from torch.optim.optimizer import Optimizer, required

class OLNM(Optimizer):
    def __init__(self, params, 
                 lr=required, 
                 T=100,
                 batch_size=required, 
                 ):

        defaults = dict(
            lr=lr, 
            batch_size=batch_size, 
            T=float(T)
        )
        
        super(OLNM, self).__init__(params, defaults)
        self.reset = True

        for group in self.param_groups:
            group['t'] = 0 # Init the small t
            group['a'] = 1.0 # Init the alpha_t
            
            for p in group['params']:
                state = self.state[p]
                state['y'] = p.data.clone().detach() # Init the y_0 = x_0
                state['z'] = p.data.clone().detach() # Init the z_0 = x_0

    def step(self, closure=None):
        x_backup = {} # Store original x data before gradient computation at y
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None: # Clear grad if grad is not None
                    p.grad.zero_()
                if 'y' in self.state[p]: # Swap to y for grad computation
                    state = self.state[p]
                    x_backup[p] = p.data.clone().detach() # Store original x data
                    p.data.copy_(state['y']) # Swap to y for grad computation
        
        # Compute loss
        loss = closure()

        # Update parameters
        group = self.param_groups[0]
        z_next_list = {} # Store z_next

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: # Restore original x data if no grad
                    if p in x_backup:
                        p.data.copy_(x_backup[p])
                    continue
                # Compute z_next = y_t - lr * grad(y_t)
                z_next = torch.add(self.state[p]['y'], p.grad.data, alpha=-group['lr'])
                z_next_list[p] = z_next # Store z_next
                # Restore original x data after gradient computation at y
                p.data.copy_(x_backup[p]) # Restore original x data

        # If T reached max, restart and update new sample
        if (group['t'] + 1) >= int(group['T']): 
            for group in self.param_groups:
                group['t'] = 0 # Restart: set t=0
                group['a'] = 1.0 # Restart: set a=1

                for p in group['params']:
                    if p not in z_next_list: # Skip if no z_next
                        continue
                    z_next = z_next_list[p] # Get z_next
                    
                    self.state[p]['y'].copy_(z_next) # Update y_t = z_next
                    self.state[p]['z'].copy_(z_next) # Update z_t = z_next
                    p.data.copy_(z_next) # Update x_t = z_next
            self.reset = True # Update new sample
           
        # Since T not reached max, update the y_t and z_t
        else:
            # Let alpha_{t+1} = (1 + sqrt(1 + 4 * alpha_t^2)) / 2
            a_next = (1 + math.sqrt(1 + 4 * group['a']**2)) / 2 
            # Let beta_t = (alpha_t - 1) / alpha_{t+1}
            beta = (group['a'] - 1) / a_next 
            
            for group in self.param_groups: # Update y_t and z_t
                for p in group['params']:
                    if p not in z_next_list: continue # Skip if no z_next
                    z_next = z_next_list[p] # Get z_next
                    z_current = self.state[p]['z'] # Get z_current
                    # Compute y_next = z_next + beta * (z_next - z_current)
                    y_next = torch.add(z_next, z_next - z_current, alpha=beta) 
                    
                    self.state[p]['z'].copy_(z_next) # Update z_t = z_next
                    self.state[p]['y'].copy_(y_next) # Update y_t = y_next
            
            group['a'] = a_next # Update alpha_{t+1}
            group['t'] += 1 # Increment t by 1
            self.reset = False # Use the same sample again

        return loss
