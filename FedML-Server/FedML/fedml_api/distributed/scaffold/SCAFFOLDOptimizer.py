from torch.optim import Optimizer

class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        pass

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group, c, ci in zip(self.param_groups, server_controls, client_controls):
            p = group['params'][0]
            if p.grad is None:
                continue
            d_p = p.grad.data + c.data - ci.data
            p.data = p.data - d_p.data * group['lr']
        # for group in self.param_groups:
        #     for p, c, ci in zip(group['params'], server_controls, client_controls):
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad.data + c.data - ci.data
        #         p.data = p.data - d_p.data * group['lr']
        return loss
