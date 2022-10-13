"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import sys
import torch
# Import paddle if using backend paddle
import paddle
# paddle.enable_static()
# paddle.incubate.autograd.enable_prim()
from deepxde.backend import backend_name, tf, torch, jax, paddle
from deepxde import backend as bkd
bkd.control_seed(100)

def ode(t, y):
    dy_dt = dde.grad.jacobian(y, t)
    d2y_dt2 = dde.grad.hessian(y, t)
    return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t


def func(t):
    return 50 / 81 + t * 5 / 9 - 2 * np.exp(t) + (31 / 81) * np.exp(9 * t)


geom = dde.geometry.TimeDomain(0, 0.25)


def boundary_l(t, on_initial):
    return on_initial and np.isclose(t[0], 0)


def bc_func1(inputs, outputs, X):
    return outputs + 1


def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2


ic1 = dde.icbc.IC(geom, lambda x: -1, lambda _, on_initial: on_initial)
ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

data = dde.data.TimePDE(geom, ode, [ic1, ic2], 16, 2, solution=func, num_test=500)

layer_size = [1] + [50] * 3 + [1]
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
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[0.01, 1, 1]
)
losshistory, train_state = model.train(iterations=10000)

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