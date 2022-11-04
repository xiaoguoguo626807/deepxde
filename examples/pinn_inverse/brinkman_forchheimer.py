"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Implementation of Brinkman-Forchheimer equation example in paper https://arxiv.org/pdf/2111.02801.pdf.
"""
import re

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
from deepxde import backend as bkd

import argparse
import paddle
import random
paddle.seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--static', default=False, action="store_true")
parser.add_argument(
    '--prim', default=False, action="store_true")
args = parser.parse_args()

if args.static is True:
    print("============= 静态图静态图静态图静态图静态图 =============")
    paddle.enable_static()
    if args.prim:
        paddle.incubate.autograd.enable_prim()
        print("============= prim prim prim prim prim  =============")
else:
    print("============= 动态图动态图动态图动态图动态图 =============")


task_name = os.path.basename(__file__).split(".")[0]

# 创建任务日志文件夹
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)
g = 1
v = 1e-3
e = 0.4
H = 1

v_e = dde.Variable(0.1)
K = dde.Variable(0.1)


def sol(x):
    r = (v * e / (1e-3 * 1e-3)) ** (0.5)
    return g * 1e-3 / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
    yvals = sol(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def pde(x, y):
    du_xx = dde.grad.hessian(y, x)
    return -v_e / e * du_xx + v * y / K - g


def output_transform(x, y):
    return x * (1 - x) * y


geom = dde.geometry.Interval(0, 1)
ob_x, ob_u = gen_traindata(5)
observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)

data = dde.data.PDE(
    geom,
    pde,
    solution=sol,
    bcs=[observe_u],
    num_domain=100,
    num_boundary=0,
    train_distribution="uniform",
    num_test=500,
)

net = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform", task_name)

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

net.apply_output_transform(output_transform)
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=[v_e, K])
variable = dde.callbacks.VariableValue([v_e, K], period=200, filename="variables1.dat")

losshistory, train_state = model.train(iterations=30000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

lines = open("variables1.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = vkinfer.shape
v_etrue = 1e-3
ktrue = 1e-3

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(vkinfer[:, 0].shape) * v_etrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), vkinfer[:, 0], "b--", label="Pred")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(top=1e-1)
plt.legend(frameon=False)
plt.ylabel(r"$\nu_e$")

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(vkinfer[:, 1].shape) * ktrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), vkinfer[:, 1], "b--", label="Pred")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(ymax=1e-1)
plt.legend(frameon=False)
plt.ylabel(r"$K$")

plt.show()
