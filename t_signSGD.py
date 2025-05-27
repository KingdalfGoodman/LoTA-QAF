import torch
from typing import List, Optional
from torch.optim.optimizer import Optimizer
from typing import Optional
# from peft.tuners.lora.layer import IntLinear
# import int_signSGD_cuda
#  ------------------------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

class tSignSGD(Optimizer):
    def __init__(
            self,
            params,
            lr=1e-5,  
            # weight_decay=0,
            threshold_ratio=0.95,   
            min_grad=0.999,
            filter_upper=0.9999,
            *,                  
            maximize=False,     
            foreach: Optional[bool] = None,  
            differentiable=False, 
            int_params=None
    ):
        if threshold_ratio < 0.0:
            raise ValueError("Invalid threshold_ratio value: {}".format(threshold_ratio))
        defaults = dict(lr=lr, maximize=maximize, foreach=foreach, differentiable=differentiable,  # weight_decay=weight_decay,
                        threshold_ratio=threshold_ratio, min_grad=min_grad, filter_upper=filter_upper,)

        super(tSignSGD, self).__init__(int_params, defaults)
        print(f"INT params count: {len(int_params)}")
        self.total_steps = None 
        self.current_step = 0

        '''
            threshold_ratio is Sigma_t, here 0.95 is discard 0.95 and select top 0.05. 
            The top 0.05 use to update weights of Ternary Adapter (TA);
        '''
        self.threshold_ratio = threshold_ratio
        self.min_grad = min_grad
        self.filter_upper = filter_upper

        
    def create_scheduler(self, num_training_steps, optimizer=None):
        self.total_steps = num_training_steps   
        self.lr_scheduler = None                

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("threshold_ratio", 0.95)   # Set default threshold_ratio
            group.setdefault("min_grad", 0.999)         # Set default min_grad
            group.setdefault("filter_upper", 0.9999)    # Set default filter_upper


    def check_grad_norm(self, threshold):
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm += torch.sum(p.grad ** 2)
        grad_norm = torch.sqrt(grad_norm).item()
        print(f"Grad norm: {grad_norm}", end='#  ')
        
        if grad_norm > threshold:
            print(f"{self.current_step} upper_banned", end='#  ')
            return True
        return False


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()  
        self.current_step += 1

        if self.total_steps is not None and self.total_steps > 0:
            upper = False
            phase1_end = self.total_steps * 0.8
            '''
                phase 1: 0-0.8 epoch  => threshold_ratio 0.95->0.999
                phase 2: 0.8-1 epoch  => threshold_ratio 0.999->0.9999
            '''

            if self.current_step <= phase1_end:
                progress = self.current_step / phase1_end
                sigma = self.threshold_ratio + progress * (self.min_grad - self.threshold_ratio)    # .x99 - .95
            else:
                upper = self.check_grad_norm(10)
                progress = (self.current_step - phase1_end) / (self.total_steps - phase1_end)

                sigma = self.min_grad + progress * (self.filter_upper - self.min_grad)              # .x999 - .x99
                if self.current_step >= self.total_steps:
                    sigma = self.filter_upper


        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                sigma_t = torch.quantile(p.grad.abs(), q=sigma)

                if upper == False:  # 1e-9 is Tau in Paper.
                    p.data.add_(torch.where((p.grad.abs() > sigma_t) & (p.grad.abs() > 1e-9), -torch.sign(p.grad), torch.zeros_like(p.grad)))
                else:
                    upper_threshold = torch.quantile(p.grad.abs(), q=0.99995)
                    p.data.add_(torch.where((p.grad.abs() > sigma_t) & (p.grad.abs() > 1e-9) & (p.grad.abs() < upper_threshold), -torch.sign(p.grad), torch.zeros_like(p.grad)))
                    # Trick: the max_grad if abnormally large, for top 0.00005 part, discard, to prevent excessive change to current parameter distribution.

                p.data.clamp_(min=-1, max=1).round_()  
        return loss

