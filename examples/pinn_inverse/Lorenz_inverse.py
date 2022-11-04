"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
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

def gen_traindata():
    data = np.load("/home/wangruting/science/deepxde_wrt_44_orig/deepxde_wrt_44/examples/dataset/Lorenz.npz")
    return data["t"], data["y"]


C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)

# Most backends
def Lorenz_system(x, y):
    """Lorenz system.
    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (15 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + C3 * y3,
    ]


# Backend JAX
# def Lorenz_system(x, y, unknowns=[C1, C2, C3]):
#     C1, C2, C3 = unknowns
#     y_val, y_fn = y
#     y1, y2, y3 = y_val[:, 0:1], y_val[:, 1:2], y_val[:, 2:3]
#     dy1_x, _ = dde.grad.jacobian(y, x, i=0)
#     dy2_x, _ = dde.grad.jacobian(y, x, i=1)
#     dy3_x, _ = dde.grad.jacobian(y, x, i=2)
#     return [
#         dy1_x - C1 * (y2 - y1),
#         dy2_x - y1 * (C2 - y3) + y2,
#         dy3_x - y1 * y2 + C3 * y3,
#     ]


def boundary(_, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(0, 3)

# Initial conditions
ic1 = dde.icbc.IC(geom, lambda X: -8, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: 7, boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: 27, boundary, component=2)

# Get the train data
observe_t, ob_y = gen_traindata()
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform", task_name)

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
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3])
variable = dde.callbacks.VariableValue(
    [C1, C2, C3], period=600, filename="variables.dat"
)
losshistory, train_state = model.train(iterations=60000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
