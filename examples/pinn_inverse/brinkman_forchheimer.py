"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Implementation of Brinkman-Forchheimer equation example in paper https://arxiv.org/pdf/2111.02801.pdf.
"""
import re

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
# Import paddle if using backend paddle
import sys
import paddle
# paddle.enable_static()
# paddle.incubate.autograd.enable_prim()
from deepxde.backend import backend_name, tf, torch, jax, paddle
from deepxde import backend as bkd
bkd.control_seed(100)

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

layer_size = [1] + [20] * 3 + [1]
# paddle init param
w_array = []
b_array = []
input_str = []
file_name1 = sys.argv[1]
with open(file_name1, mode='r') as f1:
    for line in f1:
        input_str.append(line)


j = 0
for i in range(1, len(layer_size)):
    shape_weight = (layer_size[i-1], layer_size[i])
    w_line = input_str[j]
    w = []
    tmp = w_line.split(',')
    for num in tmp:
        w.append(np.float(num))
    w = np.array(w).reshape(shape_weight)
    w_array.append(w)
    print("w . shape :", w.shape)

    j = j+1
    bias_weight = (layer_size[i])
    b_line = input_str[j]
    b = []
    tmp = b_line.split(',')
    for num in tmp:
        b.append(np.float(num))
    b = np.array(b).reshape(bias_weight)
    b_array.append(b)
    print("b . shape :", b.shape)
    j = j+1
###############################

net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform", w_array, b_array)
net.apply_output_transform(output_transform)
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=[v_e, K])
variable = dde.callbacks.VariableValue([v_e, K], period=200, filename="variables1.dat")

losshistory, train_state = model.train(iterations=100000, callbacks=[variable])
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

#########predict solution#########
x = geom.uniform_points(1000, True)
y = model.predict(x, operator=None)

if backend_name == 'paddle':
    file_namex = 'paddle_x'
    file_namey = 'paddle_y'
elif backend_name == 'pytorch':
    file_namex = 'pytorch_x'
    file_namey = 'pytorch_y'
elif backend_name == 'tensorflow':
    file_namex = 'tensorflow_x'
    file_namey = 'tensorflow_y'

    
with open(file_namex,'ab') as f:
    np.savetxt(f,x,delimiter=",")
with open(file_namey,'ab') as g:
    np.savetxt(g,y,delimiter=",")
    
# plt.figure()
# plt.plot(
#     range(0, 200 * l, 200),
#     np.ones(vkinfer[:, 0].shape) * v_etrue,
#     color="black",
#     label="Exact",
# )
# plt.plot(range(0, 200 * l, 200), vkinfer[:, 0], "b--", label="Pred")
# plt.xlabel("Epoch")
# plt.yscale("log")
# plt.ylim(top=1e-1)
# plt.legend(frameon=False)
# plt.ylabel(r"$\nu_e$")

# plt.figure()
# plt.plot(
#     range(0, 200 * l, 200),
#     np.ones(vkinfer[:, 1].shape) * ktrue,
#     color="black",
#     label="Exact",
# )
# plt.plot(range(0, 200 * l, 200), vkinfer[:, 1], "b--", label="Pred")
# plt.xlabel("Epoch")
# plt.yscale("log")
# plt.ylim(ymax=1e-1)
# plt.legend(frameon=False)
# plt.ylabel(r"$K$")

# plt.show()
