"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np

from deepxde.config import set_random_seed
from paddle.fluid import core
import paddle
import argparse
import os

set_random_seed(100)
paddle.jit.enable_to_static(False)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--static', default=False, action="store_true")
parser.add_argument(
    '--jit', default=False, action="store_true")
args = parser.parse_args()

if args.static is True:
    print("============= 静态图静态图静态图静态图静态图 =============")
    paddle.enable_static()
elif args.jit:
    paddle.jit.enable_to_static(True)
    print("============= 动转静动转静动转静动转静动转静 =============")
else:
    print("============= 动态图动态图动态图动态图动态图 =============")
if (core._is_bwd_prim_enabled()):
    print("============= 组合算子 组合算子 =============")



task_name = os.path.basename(__file__).split(".")[0]
# 创建任务日志文件夹
log_dir = f"./{task_name}"
os.makedirs(f"{log_dir}", exist_ok=True)

def ddy(x, y):
    return dde.grad.hessian(y, x)


def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + 1


def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)


def func(x):
    return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4


geom = dde.geometry.Interval(0, 1)

bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)
bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=10,
    num_boundary=2,
    solution=func,
    num_test=100,
)
layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# new_save = False
# for name, param in net.named_parameters():
#     if os.path.exists(f"{log_dir}/{name}.npy"):
#         continue
#     new_save = True
#     np.save(f"{log_dir}/{name}.npy", param.numpy())
#     print(f"successfully save param {name} at [{log_dir}/{name}.npy]")

# if new_save:
#     print("第一次保存模型完毕，自动退出，请再次运行")
#     exit(0)
# else:
#     print("所有模型参数均存在，开始训练...............")


model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
