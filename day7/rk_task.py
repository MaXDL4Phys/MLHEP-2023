### Use the adjoint method to compute the derivative of the 0-th component of the final point of the trajectory with the respect to the initial conditions for your solution of Task 0.

import torch
import torchode as to
from typing import Callable, Tuple
import torch
# import torchode
import torchdiffeq
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

# from torchdiffeq import odeint

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")

def sin_t(t, x):
    return torch.sin(t)


x_0 = torch.([12.50],[0.0]).to(device)
t_0 = torch.tensor(0.).to(device)
n_steps = 1000
t_eval = torch.stack((torch.linspace(0, 5, n_steps), torch.linspace(3, 4, n_steps)))
t_step = torch.tensor(1e-2)
term = to.ODETerm(sin_t)
step_method = to.Dopri5(term=term)

step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)

jit_solver = torch.compile(solver).cpu()

sol = jit_solver.solve(to.InitialValueProblem(y0=x_0, t_start=t_0, t_eval=t_eval))

# sol = sol.to('cpu')

# t_0 = torch.tensor(0.).to(device)
# t_step = torch.tensor(1e-2).to(device)


# with torch.no_grad():
#     x_predicted, t = adjoints(sin_t, x_0, t_0, t_step, n_steps)
#     true_x = x_0 + 1 - torch.cos(t)
#     mse = torch.nn.functional.mse_loss(x_predicted, true_x)

t = sol.t
x_predicted = sol.y
true_x = x_0 + 1 - torch.cos(t)
mse = torch.nn.functional.mse_loss(x_predicted, true_x)

fig, ax = plt.subplots()
ax.plot(t.detach().cpu(), true_x.detach().cpu(), label="analytical", lw=3)
ax.plot(t.detach().cpu(), x_predicted.detach().cpu(), label="odeint_rk4", ls="-.", lw=3)
ax.set_title(f"MSE = {mse.detach().cpu():.3}")
ax.legend()

if mse.detach().cpu() > 1e-4:
    raise ValueError("odeint_rk4 answer is too far from the ground truth")

# if mse.detach().cpu() > 1e-4:
#     raise ValueError("odeint_rk4 answer is too far from the reference")


