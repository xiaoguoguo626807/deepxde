"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import paddle

import os
import numpy as np
from deepxde.config import set_random_seed
from deepxde.backend import backend_name

set_random_seed(100)

task_name = os.path.basename(__file__).split(".")[0]
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)

def pde(x, y):
    #dy_xx = dde.grad.jacobian(y, x, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
# new_save = False
# for name, param in net.named_parameters():
#     if os.path.exists(f"{log_dir}/{name}.npy"):
#         continue
#     new_save = True
#     np.save(f"{log_dir}/{name}.npy", param.numpy())
#     print(f"successfully save param {name} at [{log_dir}/{name}.npy]")

# if new_save:
#     print("初始模型参数保存完毕，自动退出，请再次运行")
# else:
#     print("所有模型参数均存在，开始训练...............")

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
model.train(iterations=500)

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
        print("adam优化后模型参数保存完毕")
    else:
        print("所有模型参数均存在，开始训练...............")
else:
    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform", task_name)
    model = dde.Model(data, net)

model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
