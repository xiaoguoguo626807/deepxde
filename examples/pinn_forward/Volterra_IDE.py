"""Backend supported: tensorflow.compat.v1, paddle
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import deepxde.backend as bkd
from deepxde.backend import backend_name
import paddle

import os
import numpy as np
from deepxde.config import set_random_seed
set_random_seed(100)

task_name = os.path.basename(__file__).split(".")[0]
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)

def ide(x, y, int_mat):
    int_mat = bkd.as_tensor(int_mat)
    rhs = bkd.matmul(int_mat, y)
    lhs1 = bkd.gradients(y, x)[0]
    return (lhs1 + y)[: bkd.size(rhs)] - rhs


def kernel(x, s):
    return np.exp(s - x)


def func(x):
    return np.exp(-x) * np.cosh(x)


geom = dde.geometry.TimeDomain(0, 5)
ic = dde.icbc.IC(geom, func, lambda _, on_initial: on_initial)

quad_deg = 20
data = dde.data.IDE(
    geom,
    ide,
    ic,
    quad_deg,
    kernel=kernel,
    num_domain=10,
    num_boundary=2,
    train_distribution="uniform",
)
print("*********************")
layer_size = [1] + [20] * 3 + [1]

# paddle init param
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

# net = dde.nn.FNN(layer_size, activation, initializer)
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
model.compile("L-BFGS")
model.train()

# model.compile("adam", lr=1e-3)
# model.train(iterations=1)

X = geom.uniform_points(100)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

plt.figure()
plt.plot(X, y_true, "-")
plt.plot(X, y_pred, "o")
plt.show()
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
