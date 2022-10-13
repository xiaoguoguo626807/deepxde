"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import tf
import torch
# Import paddle if using backend paddle
import sys
import paddle
# paddle.enable_static()
# paddle.incubate.autograd.enable_prim()
from deepxde.backend import backend_name, tf, torch, jax, paddle
from deepxde import backend as bkd
bkd.control_seed(100)

C = dde.Variable(2.0)

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return (
        dy_t
        - C * dy_xx
        + bkd.exp(-x[:, 1:])
        * (bkd.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * bkd.sin(np.pi * x[:, 0:1]))
    )


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

observe_x = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 1))).T
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    anchors=observe_x,
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
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

model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C
)

variable = dde.callbacks.VariableValue(C, period=1000,filename="variables1.dat")
losshistory, train_state = model.train(iterations=50000, callbacks=[variable])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#########predict solution#########
x = geom.uniform_points(1000, True)
x = np.vstack((x.T, np.full((1000), 1))).T

y_ = func(x)
file_name_y_ = 'standard_y'
with open(file_name_y_,'w') as f:
    np.savetxt(f,y_,delimiter=",")

# y = model.predict(x, operator=None)

# if backend_name == 'paddle':
#     file_namex = 'paddle_x'
#     file_namey = 'paddle_y'
# elif backend_name == 'pytorch':
#     file_namex = 'pytorch_x'
#     file_namey = 'pytorch_y'
# elif backend_name == 'tensorflow':
#     file_namex = 'tensorflow_x'
#     file_namey = 'tensorflow_y'

    
# with open(file_namex,'ab') as f:
#     np.savetxt(f,x,delimiter=",")
# with open(file_namey,'ab') as g:
#     np.savetxt(g,y,delimiter=",")