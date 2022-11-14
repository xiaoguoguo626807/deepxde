"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import deepxde.backend as bkd

from deepxde.config import set_random_seed
set_random_seed(48)
import os
task_name = os.path.basename(__file__).split(".")[0]
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return (
        dy_t
        - dy_xx
        + bkd.exp(-x[:, 1:])
        * (bkd.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * bkd.sin(np.pi * x[:, 0:1]))
    )


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(geomtime, pde, [], num_domain=40, solution=func, num_test=10000)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer, task_name)

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

net.apply_output_transform(
    lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + bkd.sin(np.pi * x[:, 0:1])
)

model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
