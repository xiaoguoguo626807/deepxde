"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np

import sys
import torch
# Import paddle if using backend paddle
import paddle
paddle.enable_static()
paddle.incubate.autograd.enable_prim()
from deepxde.backend import backend_name, tf, torch, jax, paddle
from deepxde import backend as bkd
bkd.control_seed(100)


def ode_system(x, y):
    """ODE system.
    dy1/dx = y2
    dy2/dx = -y1
    """
    # Most backends
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    # Backend jax
    # y_val, y_fn = y
    # y1, y2 = y_val[:, 0:1], y_val[:, 1:]
    # dy1_x, _ = dde.grad.jacobian(y, x, i=0)
    # dy2_x, _ = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]


def boundary(_, on_initial):
    return on_initial


def func(x):
    """
    y1 = sin(x)
    y2 = cos(x)
    """
    return np.hstack((np.sin(x), np.cos(x)))


geom = dde.geometry.TimeDomain(0, 10)
ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [2]
# paddle init param
w_array = []
input_str = []
file_name1 = sys.argv[1]
with open(file_name1, mode='r') as f1:
    for line in f1:
        input_str.append(line)
print("input_str.size: ", len(input_str))
j = 0
for i in range(1, len(layer_size)):
    shape = (layer_size[i-1], layer_size[i])
    w_line = input_str[j]
    w = []
    tmp = w_line.split(',')
    for num in tmp:
        w.append(np.float(num))
    w = np.array(w).reshape(shape)
    print("w . shape :", w.shape)
    j = j+2
    w_array.append(w)
###############################
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer, w_array)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x = geom.uniform_points(1000, True)
y_ = func(x)
file_name_y_ = 'standard_y'
with open(file_name_y_,'w') as f:
    np.savetxt(f,y_,delimiter=",")
    
# y = model.predict(x, operator=pde)
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