"""Backend supported: tensorflow.compat.v1, tensorflow"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import deepxde.backend as bkd
from deepxde.backend import backend_name
import paddle
from scipy.interpolate import griddata

import os
import numpy as np
from deepxde.config import set_random_seed
set_random_seed(100)

task_name = os.path.basename(__file__).split(".")[0]
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def pde(x, y):
    alpha, beta, gamma, k = -1, 0, 1, 2
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    x, t = x[:, 0:1], x[:, 1:2]
    
    return (
        dy_tt
        + alpha * dy_xx
        + beta * y
        + gamma * (y ** k)
        + x * bkd.cos(t)
        - (x ** 2) * (bkd.cos(t) ** 2)
    )

def func(x):
    return x[:, 0:1] * np.cos(x[:, 1:2])


bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic_1 = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
ic_2 = dde.icbc.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda _, on_initial: on_initial,
)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic_1, ic_2],
    num_domain=30000,
    num_boundary=1500,
    num_initial=1500,
    solution=func,
    num_test=6000,
)

layer_size = [2] + [40] * 2 + [1]

# w_array = []
# if backend_name == "tensorflow":
#     input_str = []
#     import sys
#     file_name1 = sys.argv[1]
#     with open(file_name1, mode='r') as f1:
#         for line in f1:
#             input_str.append(line)
#     print("input_str.size: ", len(input_str))
#     j = 0
#     for i in range(1, len(layer_size)):
#         shape = (layer_size[i-1], layer_size[i])
#         w_line = input_str[j]
#         w = []
#         tmp = w_line.split(',')
#         for num in tmp:
#             w.append(np.float(num))
#         w = np.array(w).reshape(shape)
#         print("w . shape :", w.shape)
#         j = j+2
#         w_array.append(w)


activation = "tanh"
initializer = "Glorot uniform"

net = dde.nn.FNN(layer_size, activation, initializer, task_name)

from deepxde.backend import backend_name
if backend_name == 'paddle':
    new_save = False
    for name, param in net.named_parameters():
        if os.path.exists(f"{log_dir}/{name}.npy"):
            continue
        new_save = True
        np.save(f"{log_dir}/{name}.npy", param.numpy())
        print(f"successfully save param {name} at [{log_dir}/{name}.npy]")

    if new_save:
        print("第一次保存模型完毕，自动退出，请再次运行")
        exit(0)
    else:
        print("所有模型参数均存在，开始训练...............")


model = dde.Model(data, net)
model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", 3000, 0.9)
)
model.train(iterations=20000)
model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x = np.linspace(-1, 1, 256)
t = np.linspace(0, 10, 256)
X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
prediction = model.predict(X_star, operator=None)

v = griddata(X_star, prediction[:, 0], (X, T), method="cubic")

fig, ax = plt.subplots()
ax.set_title("Results")
ax.set_ylabel("Prediciton")
ax.imshow(
    v.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[0, 10, -1, 1],
    origin="lower",
    aspect="auto",
)
plt.show()
