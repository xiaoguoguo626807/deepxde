"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde

import os
import numpy as np
from deepxde.config import set_random_seed
set_random_seed(100)

task_name = os.path.basename(__file__).split(".")[0]
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)


Re = 20
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)


def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]


def u_func(x):
    return 1 - np.exp(l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])


def v_func(x):
    return l / (2 * np.pi) * np.exp(l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2])


def p_func(x):
    return 1 / 2 * (1 - np.exp(2 * l * x[:, 0:1]))


def boundary_outflow(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


spatial_domain = dde.geometry.Rectangle(xmin=[-0.5, -0.5], xmax=[1, 1.5])

boundary_condition_u = dde.icbc.DirichletBC(
    spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.icbc.DirichletBC(
    spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_right_p = dde.icbc.DirichletBC(
    spatial_domain, p_func, boundary_outflow, component=2
)

data = dde.data.TimePDE(
    spatial_domain,
    pde,
    [boundary_condition_u, boundary_condition_v, boundary_condition_right_p],
    num_domain=2601,
    num_boundary=400,
    num_test=100000,
)

net = dde.nn.FNN([2] + 4 * [50] + [3], "tanh", "Glorot normal", task_name)

from deepxde.backend import backend_name
if backend_name == 'pytorch':
    new_save = False
    i = 0
    for name, param in net.named_parameters():
        if os.path.exists(f"{log_dir}/{name}.npy"):
            continue
        new_save = True
        if i % 2 == 0:
            np.save(f"{log_dir}/{name}.npy", np.transpose(param.cpu().detach().numpy()))
        else:
            np.save(f"{log_dir}/{name}.npy", param.cpu().detach().numpy())
        print(f"successfully save param {name} at [{log_dir}/{name}.npy]")
        i += 1
    if new_save:
        print("初始化模型参数保存完毕")
        exit(0)
    else:
        print("所有模型参数均存在，开始训练...............")

# new_save = False
# for name, param in net.named_parameters():
#     if os.path.exists(f"{log_dir}/{name}.npy"):
#         continue
#     new_save = True
#     np.save(f"{log_dir}/{name}.npy", param.numpy())
#     print(f"successfully save param {name} at [{log_dir}/{name}.npy]")

# if new_save:
#     print("第一次保存模型完毕，自动退出，请再次运行")
#     exit(0)
# else:
#     print("所有模型参数均存在，开始训练...............")


model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(iterations=30000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

X = spatial_domain.random_points(100000)
output = model.predict(X)
u_pred = output[:, 0]
v_pred = output[:, 1]
p_pred = output[:, 2]

u_exact = u_func(X).reshape(-1)
v_exact = v_func(X).reshape(-1)
p_exact = p_func(X).reshape(-1)

f = model.predict(X, operator=pde)

l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
residual = np.mean(np.absolute(f))

print("Mean residual:", residual)
print("L2 relative error in u:", l2_difference_u)
print("L2 relative error in v:", l2_difference_v)
print("L2 relative error in p:", l2_difference_p)
