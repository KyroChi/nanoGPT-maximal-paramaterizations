# Optimizer Aware Weight Decay Annealing (OWDA)
# 
# This is an AdamW version of the OWDA optimizer, i.e. OWDA-Adam.
# I also want to experiment with OWDA-MuON.

import math
import torch
from torch.optim import AdamW

class OWDAAdam(AdamW):
    def __init__(
            self, 
            params, 
            lr=1e-3, 
            weight_decay=0.01, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            amsgrad=False
        ):
        super().__init__(params, lr=lr, weight_decay=0.0, betas=betas, eps=eps, amsgrad=amsgrad)
        self.base_weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.defaults['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom_r = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom_r = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                r = (exp_avg / bias_correction1).div(denom_r)

                if p.data.norm() == 0:
                    dynamic_lambda = self.base_weight_decay
                else:
                    dynamic_lambda = self.base_weight_decay * (r.norm() / p.data.norm())

                p.data.mul_(1 - group['lr'] * dynamic_lambda)
                p.data.addcdiv_(exp_avg, denom_r, value=-group['lr'])

        return loss