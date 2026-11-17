# Reference runs of the optimizers: the same gradients fed to PyTorch's (and transformers')
# implementations for five steps, and the weights after each run.
#   python make_optimizers.py > optimizers.json
import json, torch
from transformers.optimization import Adafactor
import torch.optim._muon as muon_mod
torch.manual_seed(0)
W0 = torch.randn(4, 6)
B0 = torch.randn(6)
grads = [(torch.randn(4, 6), torch.randn(6)) for _ in range(5)]

def run(make, matrix_only=False):
    W = torch.nn.Parameter(W0.clone())
    B = torch.nn.Parameter(B0.clone())
    params = [W] if matrix_only else [W, B]
    opt = make(params)
    for gw, gb in grads:
        W.grad = gw.clone()
        B.grad = gb.clone()
        opt.step()
    return [p.detach().numpy().ravel().tolist() for p in params]

class Lion(torch.optim.Optimizer):
    # the paper's algorithm (Chen et al. 2023, Program 1)
    def __init__(self, params, lr, betas, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)

# PyTorch's Muon orthogonalizes in bfloat16; the reference keeps float32 so the comparison is tight
orig = muon_mod._zeropower_via_newtonschulz
def ns32(grad, coeffs, steps, eps):
    a, b, c = coeffs
    x = grad.float().clone()
    tall = grad.size(0) > grad.size(1)
    if tall: x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    return x.T if tall else x
muon_mod._zeropower_via_newtonschulz = ns32

out = {"W0": W0.numpy().ravel().tolist(), "B0": B0.numpy().ravel().tolist(),
       "grads": [[gw.numpy().ravel().tolist(), gb.numpy().ravel().tolist()] for gw, gb in grads]}
out["adam"] = run(lambda p: torch.optim.Adam(p, lr=0.01))
out["adamw"] = run(lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1))
out["lion"] = run(lambda p: Lion(p, lr=0.01, betas=(0.9, 0.99), weight_decay=0.1))
out["adafactor"] = run(lambda p: Adafactor(p, lr=0.01, scale_parameter=False, relative_step=False))
out["adafactor_relative"] = run(lambda p: Adafactor(p, lr=None, scale_parameter=True, relative_step=True,
                                                     weight_decay=0.01))
out["muon"] = run(lambda p: torch.optim.Muon(p, lr=0.02), matrix_only=True)
out["sgd_momentum"] = run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))
print(json.dumps(out))
