# NC2d自然对流换热案例
import paddle
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation
import numpy as np
import os

# 设置实际时空计算域的参考尺寸(仅用于后处理显示实际流场和温度场,求解过程中一律采用无量纲的计算域: x[0,1] × y[0,1] × t[0,1])
D_L = 1.0; D_H = 1.0    # 实际矩形长×宽
t_r = 1.0               # 实际计算时长
T_h = 1.0; T_c = 0.0    # 实际左右壁面温度
step_x = 0.01; step_y = 0.01; step_t = 0.1 # x, y, t采样间隔
Tref = (T_h + T_c) / 2.0# 热浮力项中的参考温度
# 设置稳态or瞬态模拟
isTransient = False         # True-瞬态模拟，False-稳态模拟
# 设置方程系数(物性参数)
rho = 1.29                  # 空气密度1.29 kg/m3, 水密度1000 kg/m3
mu = 17.9e-6                # 空气动力粘度17.9×10-6 Pa·s, 水动力粘度1.01×10-3 Pa·s
nu = mu / rho               # 空气运动粘度14.8×10-6 m2/s, 水运动粘度1.01×10-6 m2/s
beta = 3.4e-3               # 空气热膨胀系数3.4×10-3 (1/℃), 水热膨胀系数0.21×10-3 (1/℃)
k = 0.023                   # 空气导热系数0.023 W/(m·℃), 水导热系数0.59 W/(m·℃)
cp = 1.004e3                # 空气比热1.004×103 J/(kg·℃), 水比热4.2×103 J/(kg·℃)
alpha = k / (rho * cp)      # 热扩散系数m2·s, 空气的热扩散率0.000024 m2/s
gravity = 9.8               # 重力加速度m/s2
# 设置初始条件参数(采用函数来指定)
Vinit = np.array([0, 0, 0]) # 初始速度和压力
Tinit_q = 0.0               # 热源区域的初始温度
Tinit = 0.0                 # 其它区域的初始温度
# 设置热源参数
Q = 0.0                     # 能量方程源项（热源）
x_qs = np.array([
    [0.25 * D_L, 0.25 * D_H], 
    [0.25 * D_L, 0.75 * D_H], 
    [0.75 * D_L, 0.25 * D_H], 
    [0.75 * D_L, 0.75 * D_H]])                  # 热源中心坐标, 每个坐标表示一个热源
qtype = 2                                       # 0-圆形, 1-矩形, 其它-整体均匀热源Q
lw_q = np.array([0.2 * D_L, 0.2 * D_H])         # 统一的矩形热源形状参数
r_qs = np.sqrt(lw_q[0] ** 2 + lw_q[1] ** 2) / 2 # 统一的圆形热源形状参数
# 设置热边界条件参数
BCtype = 1                  # 选择热边界条件类型: 3-Robin, 2-Neumann, others-Dirichlet
Tbc_l = T_h                 # 设置Dirichlet热边界条件参数-左侧边界
Tbc_r = T_c                 # 设置Dirichlet热边界条件参数-右侧边界
Tbc_b = 0.0                 # 设置Dirichlet热边界条件参数-下侧边界
Tbc_t = 0.0                 # 设置Dirichlet热边界条件参数-上侧边界
qbc_l = 0.0                 # 设置Neumann热边界条件参数-左侧边界
qbc_r = 0.0                 # 设置Neumann热边界条件参数-右侧边界
qbc_b = 0.0                 # 设置Neumann热边界条件参数-下侧边界
qbc_t = 0.0                 # 设置Neumann热边界条件参数-上侧边界
h_l = 0.0;    Tamb_l = 0.0  # 设置Robin热边界条件参数-左侧边界
h_r = 100.0;  Tamb_r = 0.0  # 设置Robin热边界条件参数-右侧边界
h_b = 0.0;    Tamb_b = 0.0  # 设置Robin热边界条件参数-下侧边界
h_t = 0.0;    Tamb_t = 0.0  # 设置Robin热边界条件参数-上侧边界
# ------------start构造Pr = 0.71, Ra = 1.0e3------------
k = 1.0 * rho * cp
alpha = k / (rho * cp)
nu = 0.71 * alpha
mu = nu * rho
beta = 1.0e3 * nu * alpha / (gravity * (T_h - T_c) * (D_H ** 3))
# ------------finish构造Pr = 0.71, Ra = 1.0e3------------
# 计算无量纲参数
Pr = nu / alpha                                             # 普朗特数Pr
Gr = gravity * beta * (T_h - T_c) * (D_H ** 3) / (nu ** 2)  # 格拉晓夫数Gr
Ra = Pr * Gr                                                # 瑞利数Ra，可测试：10^3, 10^4, 10^5, 10^6, 10^7, 10^8
# 计算无量纲参考参数
t_star = D_H * D_H / alpha
xy_star = D_H
uv_star = alpha / D_H
p_star = rho * uv_star * uv_star
T_star = T_h - T_c
Q_star = k * (T_h - T_c) / (D_H ** 2)
# 计算无量纲时空计算域(用于求解设置)
L_d = D_L / xy_star
H_d = D_H / xy_star
t_d = t_r / t_star
# 计算无量纲的局部热源参数
Q_d = Q / Q_star
lw_q_d = lw_q / xy_star
r_qs_d = r_qs / xy_star
# 计算无量纲初始条件参数
Vinit_d = np.array([0, 0, 0])           # 初始速度和压力
Vinit_d[0] = Vinit[0] / uv_star
Vinit_d[1] = Vinit[1] / uv_star
Vinit_d[2] = Vinit[2] / p_star
Tinit_d_q = (Tinit_q - Tref) / T_star    # 热源区域的初始温度
Tinit_d = (Tinit - Tref) / T_star        # 其它区域的初始温度
# 计算无量纲边界条件参数
Tbc_d_l = (Tbc_l - Tref) / T_star                        # 设置Dirichlet热边界条件参数-左侧边界
Tbc_d_r = (Tbc_r - Tref) / T_star                        # 设置Dirichlet热边界条件参数-右侧边界
Tbc_d_b = (Tbc_b - Tref) / T_star                        # 设置Dirichlet热边界条件参数-下侧边界
Tbc_d_t = (Tbc_t - Tref) / T_star                        # 设置Dirichlet热边界条件参数-上侧边界
dTbc_d_l = - qbc_l / k * xy_star / T_star                   # 设置Neumann热边界条件参数-左侧边界
dTbc_d_r = - qbc_r / k * xy_star / T_star                   # 设置Neumann热边界条件参数-右侧边界
dTbc_d_b = - qbc_b / k * xy_star / T_star                   # 设置Neumann热边界条件参数-下侧边界
dTbc_d_t = - qbc_t / k * xy_star / T_star                   # 设置Neumann热边界条件参数-上侧边界
h_d_l = h_l * xy_star / k;  Tamb_d_l = (Tamb_l - Tref) / T_star  # 设置Robin热边界条件参数-左侧边界
h_d_r = h_r * xy_star / k;  Tamb_d_r = (Tamb_r - Tref) / T_star  # 设置Robin热边界条件参数-右侧边界
h_d_b = h_b * xy_star / k;  Tamb_d_b = (Tamb_b - Tref) / T_star  # 设置Robin热边界条件参数-下侧边界
h_d_t = h_t * xy_star / k;  Tamb_d_t = (Tamb_t - Tref) / T_star  # 设置Robin热边界条件参数-上侧边界

# 计算采样点数
num_points_x = int(D_L / step_x)
num_points_y = int(D_H / step_y)
num_points_t = int(t_r / step_t)
if not isTransient:
    num_points_t = 1

# ----------------start总结输入参数------------------
print('NaturalConvection2d: Pr = ', Pr, ', Gr = ', Gr, ', Ra = ', Ra)
print('NaturalConvection2d: L_d = ', L_d, ', H_d = ', H_d, ', t_d = ', t_d)
print('NaturalConvection2d: Tbc_d_l = ', Tbc_d_l, ', Tbc_d_r = ', Tbc_d_r)
print('NaturalConvection2d: t_star = ', t_star, ', xy_star = ', xy_star, ', uv_star = ', uv_star, 
    ', p_star = ', p_star, ', T_star = ', T_star, ', Q_star = ', Q_star)
# exit(0)
# ----------------finish总结输入参数------------------

# The "physics-informed" part of the loss
if isTransient:
    time_label = "transient"
else:
    time_label = "steady"
f_prefix = 'NC2d_' + str(int(Ra)) + '_' + time_label + '_BC' + str(BCtype)
work_path = os.path.join('results', f_prefix,)
isCreated = os.path.exists(work_path)
if not isCreated:
    os.makedirs(work_path)
print("保存路径: " + work_path)

# 定义无量纲时空计算域
geom = dde.geometry.Rectangle([0, 0], [L_d, H_d])
if isTransient:
    timedomain = dde.geometry.TimeDomain(0, t_d)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
else:
    geomtime = geom

def sourceQ(x, y):
    if qtype == 0:
        # 多圆形热源区域
        Qx = paddle.zeros([x.shape[0],])
        mask = paddle.full(shape=[x.shape[0],x_qs.shape[0]], fill_value=True, dtype='bool')
        for ii in range(0, x_qs.shape[0]):
            mask[:,ii] = paddle.logical_and( mask[:,ii], 
            paddle.sqrt( (x[:,0]-x_qs[ii,0])**2 + (x[:,1]-x_qs[ii,1])**2 ) <= r_qs )
        Qx[paddle.any(mask, axis=1)] = Q_d
    elif qtype == 1:
        # 多矩形热源区域
        Qx = paddle.zeros([x.shape[0],])
        mask = paddle.full(shape=[x.shape[0],x_qs.shape[0]], fill_value=False, dtype='bool')
        for ii in range(0, x_qs.shape[0]):
            mask[:,ii] = paddle.logical_and(x[:,0]-x_qs[ii,0] >= -lw_q[0]/2, x[:,0]-x_qs[ii,0] <= lw_q[0]/2)
            mask[:,ii] = paddle.logical_and(mask[:,ii], x[:,1]-x_qs[ii,1] >= -lw_q[0]/2)
            mask[:,ii] = paddle.logical_and(mask[:,ii], x[:,1]-x_qs[ii,1] <= lw_q[1]/2)
        Qx[paddle.any(mask, axis=1)] = Q_d
    else:
        # 整体均匀热源
        Qx = paddle.ones([x.shape[0],]) * Q_d
    # print('[x, Qx]: ', paddle.concat(x=[x, Qx[:,None]], axis=-1))
    return Qx[:,None]

def pde(x, y):
    """
    INPUTS:
        x: x[:,0] is x1-coordinate
           x[:,1] is x2-coordinate
           x[:,2] is t-coordinate
        y: Network output, in this case:
           y[:,0] is u(x1,x2,t)
           y[:,1] is v(x1,x2,t)
           y[:,2] is p(x1,x2,t)
           y[:,3] is T(x1,x2,t)
    OUTPUTS:
        The pde in standard form i.e. something that must be zero
    """
    # print('pde: ', type(x), x.dtype, x.shape)  # <class 'paddle.Tensor'> paddle.float32 (:, 3)
    # print('pde: ', type(y), y.dtype, y.shape)  # <class 'paddle.Tensor'> paddle.float32 (:, 4)
    # 注: 这里的x和y均为Paddle(backend).Tensor, 所以这里需要Paddle的API(比如, sourceQ中用到sin函数的API为paddle.sin)
    
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    T = y[:, 3:4]
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    dT_x = dde.grad.jacobian(y, x, i=3, j=0)
    dT_y = dde.grad.jacobian(y, x, i=3, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    dT_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)

    continuity = du_x + dv_y
    u_momentum = (u * du_x + v * du_y) + dp_x - Pr * (du_xx + du_yy)
    v_momentum = (u * dv_x + v * dv_y) + dp_y - Pr * (dv_xx + dv_yy) - T
    # v_momentum = (u * dv_x + v * dv_y) + dp_y - Pr * (dv_xx + dv_yy) - Ra * Pr * T
    # v_momentum = (u * dv_x + v * dv_y) / (Ra * Pr) + dp_y / (Ra * Pr) - (dv_xx + dv_yy) / Ra - T
    T_energy = (u * dT_x + v * dT_y) - (dT_xx + dT_yy) - sourceQ(x, y)
    if isTransient:
        du_t = dde.grad.jacobian(y, x, i=0, j=2)
        dv_t = dde.grad.jacobian(y, x, i=1, j=2)
        dT_t = dde.grad.jacobian(y, x, i=3, j=2)
        u_momentum = u_momentum + du_t
        v_momentum = v_momentum + dv_t
        T_energy = T_energy + dT_t

    # print('pde(x, y)-shape: ', continuity.shape, u_momentum.shape, v_momentum.shape, T_energy.shape) # paddle.float32 (:, 1)
    # norm2_continuity = continuity.norm(2);  print('pde(x, y)-norm2_continuity: ', norm2_continuity)
    # norm2_u_momentum = u_momentum.norm(2);  print('pde(x, y)-norm2_u_momentum: ', norm2_u_momentum)
    # norm2_v_momentum = v_momentum.norm(2);  print('pde(x, y)-norm2_v_momentum: ', norm2_v_momentum)
    # norm2_T_energy = T_energy.norm(2);      print('pde(x, y)-norm2_T_energy: ', norm2_T_energy)
    # mse_continuity = paddle.mean(paddle.square(continuity)); print('pde(x, y)-mse_continuity: ', mse_continuity)
    # mse_u_momentum = paddle.mean(paddle.square(u_momentum)); print('pde(x, y)-mse_u_momentum: ', mse_u_momentum)
    # mse_v_momentum = paddle.mean(paddle.square(v_momentum)); print('pde(x, y)-mse_v_momentum: ', mse_v_momentum)
    # mse_T_energy = paddle.mean(paddle.square(T_energy));     print('pde(x, y)-mse_T_energy: ', mse_T_energy)

    return [continuity, u_momentum, v_momentum, T_energy]

# Boundary and Initial conditions

def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], L_d)

def boundary_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

def boundary_t(x, on_boundary):
    return on_boundary and np.isclose(x[1], H_d)

def boundary_all(_, on_boundary):
    return on_boundary

def boundary_p_ref(x, on_boundary):
    return on_boundary and ( np.isclose(x[0], L_d) and np.isclose(x[1], H_d) )

# bc_u_l = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l, component=0)
# bc_u_r = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_r, component=0)
# bc_u_b = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_b, component=0)
# bc_u_t = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_t, component=0)
bc_u = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_all, component=0)
bc_v = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_all, component=1)
bc_p = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_p_ref, component=2)

if BCtype == 3:
    bc_T_l = dde.icbc.RobinBC(geomtime, lambda x,y: - h_d_l * (y[:, 3:4] - Tamb_d_l), boundary_l, component=3)
    bc_T_r = dde.icbc.RobinBC(geomtime, lambda x,y: - h_d_r * (y[:, 3:4] - Tamb_d_r), boundary_r, component=3)
    bc_T_b = dde.icbc.RobinBC(geomtime, lambda x,y: - h_d_b * (y[:, 3:4] - Tamb_d_b), boundary_b, component=3)
    bc_T_t = dde.icbc.RobinBC(geomtime, lambda x,y: - h_d_t * (y[:, 3:4] - Tamb_d_t), boundary_t, component=3)
elif BCtype == 2:
    bc_T_l = dde.icbc.NeumannBC(geomtime, lambda x: dTbc_d_l, boundary_l, component=3)
    bc_T_r = dde.icbc.NeumannBC(geomtime, lambda x: dTbc_d_r, boundary_r, component=3)
    bc_T_b = dde.icbc.NeumannBC(geomtime, lambda x: dTbc_d_b, boundary_b, component=3)
    bc_T_t = dde.icbc.NeumannBC(geomtime, lambda x: dTbc_d_t, boundary_t, component=3)
else:
    bc_T_l = dde.icbc.DirichletBC(geomtime, lambda x: Tbc_d_l, boundary_l, component=3)
    bc_T_r = dde.icbc.DirichletBC(geomtime, lambda x: Tbc_d_r, boundary_r, component=3)
    # bc_T_b = dde.icbc.DirichletBC(geomtime, lambda x: Tbc_b, boundary_b, component=3)
    # bc_T_t = dde.icbc.DirichletBC(geomtime, lambda x: Tbc_t, boundary_t, component=3)

def ic_u_func(x):
    return np.ones([x.shape[0], 1], dtype='float32') * Vinit_d[0]

def ic_v_func(x):
    return np.ones([x.shape[0], 1], dtype='float32') * Vinit_d[1]

def ic_p_func(x):
    return np.ones([x.shape[0], 1], dtype='float32') * Vinit_d[2]

def ic_T_func(x):
    # return np.sin( np.pi * (x[:,0]/Lx0 + x[:,1]/Lx1) ) # 参考写法
    # print('ic_func: ', type(x), x.dtype, x.shape) # <class 'numpy.ndarray'> float32 (10000, 3)

    if qtype == 0:
        # 多圆形热源区域
        Tic_x = np.ones([x.shape[0],]) * Tinit_d
        mask = np.full(shape=[x.shape[0],x_qs.shape[0]], fill_value=True, dtype='bool')
        for ii in range(0, x_qs.shape[0]):
            mask[:,ii] = np.logical_and( mask[:,ii], 
            np.sqrt( (x[:,0]-x_qs[ii,0])**2 + (x[:,1]-x_qs[ii,1])**2 ) <= r_qs )
        Tic_x[np.any(mask, axis=1)] = Tinit_d_q
    elif qtype == 1:
        # 多矩形热源区域
        Tic_x = np.ones([x.shape[0],]) * Tinit_d
        mask = np.full(shape=[x.shape[0],x_qs.shape[0]], fill_value=False, dtype='bool')
        for ii in range(0, x_qs.shape[0]):
            mask[:,ii] = np.logical_and(x[:,0]-x_qs[ii,0] >= -lw_q[0]/2, x[:,0]-x_qs[ii,0] <= lw_q[0]/2)
            mask[:,ii] = np.logical_and(mask[:,ii], x[:,1]-x_qs[ii,1] >= -lw_q[0]/2)
            mask[:,ii] = np.logical_and(mask[:,ii], x[:,1]-x_qs[ii,1] <= lw_q[1]/2)
        Tic_x[np.any(mask, axis=1)] = Tinit_d_q
    else:
        # 整体均匀热源
        Tic_x = np.ones([x.shape[0],]) * Tinit_d_q
    # print('[x, Tic_x]: ', np.concatenate([x, Tic_x[:,None]], axis=-1))
    return Tic_x[:,None]

if isTransient:
    ic_u = dde.icbc.IC(geomtime, ic_u_func, lambda _, on_initial: on_initial, component=0)
    ic_v = dde.icbc.IC(geomtime, ic_v_func, lambda _, on_initial: on_initial, component=1)
    ic_p = dde.icbc.IC(geomtime, ic_p_func, lambda _, on_initial: on_initial, component=2) # 测试下是否可以不初始化p
    ic_T = dde.icbc.IC(geomtime, ic_T_func, lambda _, on_initial: on_initial, component=3)

# 定义问题和模型，完成训练求解
if isTransient:
    data = dde.data.TimePDE(
        geomtime,
        pde,
        # [bc_u, bc_v, bc_p, bc_T_l, bc_T_r, ic_u, ic_v, ic_p, ic_T],
        [bc_u, bc_v, bc_T_l, bc_T_r, ic_u, ic_v, ic_p, ic_T],
        num_domain=num_points_x * num_points_y * num_points_t,                # 12000
        num_boundary=(num_points_x * 2 + num_points_y * 2) * num_points_t,    # 320
        num_initial=num_points_x * num_points_y,                              # 800
        num_test=num_points_x * num_points_y * num_points_t,                  # 12000
        # num_domain=num_points_x * num_points_y * 10,
        # num_boundary=(num_points_x * 2 + num_points_y * 2) * num_points_t,
        # num_initial=num_points_x * num_points_y,
        # num_test=num_points_x * num_points_y * 10,
    )
else:
    data = dde.data.PDE(
        geomtime, 
        pde, 
        # [bc_u, bc_v, bc_p, bc_T_l, bc_T_r],
        [bc_u, bc_v, bc_T_l, bc_T_r],
        # [bc_u_l, bc_u_r, bc_u_b, bc_u_t, bc_v, bc_T_l, bc_T_r],
        # num_domain=num_points_x * num_points_y,                             # 12000
        # num_boundary=(num_points_x * 2 + num_points_y * 2),                     # 320
        # num_test=num_points_x * num_points_y,                               # 12000
        num_domain=num_points_x * num_points_y,
        num_boundary=(num_points_x * 2 + num_points_y * 2),
        num_test=num_points_x * num_points_y,
    )

if isTransient:
    layer_size = [3] + [50] * 6 + [4]
else:
    layer_size = [2] + [50] * 6 + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
# net = dde.nn.FNN([3] + [50] * 6 + [4], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss="MSE")
model.restore(save_path=work_path + "//" + f_prefix + "_model-40000.pdparams")
# losshistory, train_state = model.train(iterations=20000, display_every=1000, 
#     disregard_previous_best=False, model_restore_path=None, 
#     model_save_path=work_path + "//" + f_prefix + "_model")
# model.compile("adam", lr=1e-4, loss="MSE")
# losshistory, train_state = model.train(iterations=20000, display_every=1000, 
#     disregard_previous_best=False, model_restore_path=None, 
#     model_save_path=work_path + "//" + f_prefix + "_model")

# Plot/print the results
# dde.saveplot(losshistory, train_state, issave=True, isplot=True, 
#     loss_fname=work_path + "//" + "loss.dat",
#     train_fname=work_path + "//" + "train.dat",
#     test_fname=work_path + "//" + "test.dat")

# 所需输入: t_L, D_L, D_H, model
x1 = np.linspace(start=0, stop=L_d, num=100, endpoint=True).flatten() # (100,)
x2 = np.linspace(start=0, stop=H_d, num=100, endpoint=True).flatten() # (100,)
XX1, XX2 = np.meshgrid(x1, x2)
x_1 = XX1.flatten()
x_2 = XX2.flatten()
Nt = num_points_t
dt = t_d / Nt
if isTransient:
    for n in range(0, Nt+1):
        xt = n * dt
        xt_list = xt * np.ones((len(x_1), 1))
        x_pred = np.concatenate([x_1[:, None], x_2[:, None], xt_list], axis=1)# (num_of_points, 3)
        y_pred = model.predict(x_pred)                                        # (num_of_points, 4)
        data_n = np.concatenate([x_pred, y_pred], axis=1)
        # print("x_pred.shape: ", x_pred.shape, ", y_pred.shape: ", y_pred.shape, ", data_n.shape: ", data_n.shape)
        if n == 0:
            data = data_n[:, :, None]
        else:
            data = np.concatenate([data, data_n[:, :, None]], axis=2)         # (num_of_points, 7, Nt+1)
else:
    x_pred = np.concatenate([x_1[:, None], x_2[:, None]], axis=1)
    y_pred = model.predict(x_pred)
    xt_list = np.zeros((len(x_1), 1))
    data_n = np.concatenate([x_pred, xt_list, y_pred], axis=1)
    data = data_n[:, :, None]

print(x_pred.shape, y_pred.shape)
print(data.shape, data_n.shape)

# 所需输入: data, dt, Nt
# 获得y的最大值和最小值
u_min = data.min(axis=(0, 2,))[3]
u_max = data.max(axis=(0, 2,))[3]
v_min = data.min(axis=(0, 2,))[4]
v_max = data.max(axis=(0, 2,))[4]
p_min = data.min(axis=(0, 2,))[5]
p_max = data.max(axis=(0, 2,))[5]
T_min = data.min(axis=(0, 2,))[6]
T_max = data.max(axis=(0, 2,))[6]
print("u_min = ", u_min, ", u_max = ", u_max)
print("v_min = ", v_min, ", v_max = ", v_max)
print("p_min = ", p_min, ", p_max = ", p_max)
print("T_min = ", T_min, ", T_max = ", T_max)

# 设置colorbar显示的级别
levels_u = np.arange(u_min, u_max + (u_max - u_min) / 30, (u_max - u_min) / 30)
levels_v = np.arange(v_min, v_max + (v_max - v_min) / 30, (v_max - v_min) / 30)
levels_p = np.arange(p_min, p_max + (p_max - p_min) / 30, (p_max - p_min) / 30)
levels_T = np.arange(T_min, T_max + (T_max - T_min) / 30, (T_max - T_min) / 30)

fig = plt.figure(100, figsize=(20, 4))

def init():
    plt.clf()
    x1_t = data[:, 0:1, 0]
    x2_t = data[:, 1:2, 0]
    xt_0 = 0 * np.ones([data.shape[0], 1])
    xinit = np.concatenate([x1_t, x2_t, xt_0], axis=-1)
    u_p_t = ic_u_func(xinit)
    v_p_t = ic_v_func(xinit)
    p_p_t = ic_p_func(xinit)
    T_p_t = ic_T_func(xinit)
    # print('x1_t: ', x1_t)
    # print('x2_t: ', x2_t)
    # print('u_p_t: ', u_p_t)
    # print('v_p_t: ', v_p_t)
    # print('p_p_t: ', p_p_t)
    # print('T_p_t: ', T_p_t)

    ax1 = plt.subplot(1, 4, 1)
    print('init()-u: ', x1_t.shape, x2_t.shape, u_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), u_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), u_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(u_min, u_max):
        plt.clim(vmin=u_min, vmax=u_max * 1.0001)
    cb1 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("u-field with BCs/IC.", fontsize = 9.5)

    ax2 = plt.subplot(1, 4, 2)
    print('init()-v: ', x1_t.shape, x2_t.shape, v_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), v_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), v_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(v_min, v_max):
        plt.clim(vmin=v_min, vmax=v_max * 1.0001)
    cb2 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("v-field with BCs/IC.", fontsize = 9.5)

    ax3 = plt.subplot(1, 4, 3)
    print('init()-p: ', x1_t.shape, x2_t.shape, p_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), p_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), p_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(p_min, p_max):
        plt.clim(vmin=p_min, vmax=p_max * 1.0001)
    cb3 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("p-field with BCs/IC.", fontsize = 9.5)

    ax4 = plt.subplot(1, 4, 4)
    print('init()-T: ', x1_t.shape, x2_t.shape, T_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), T_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), T_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(T_min, T_max):
        plt.clim(vmin=T_min, vmax=T_max * 1.0001)
    cb4 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("T-field with BCs/IC.", fontsize = 9.5)

    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    # fig.tight_layout(pad=0, h_pad=.1, w_pad=.1)
    fig.tight_layout()
    # plt.show()
    plt.savefig(work_path + '//' + f_prefix + '_animation_BCsIC' + '.jpg')

def anim_update(t_id):
    plt.clf()
    x1_t = data[:, 0:1, t_id]
    x2_t = data[:, 1:2, t_id]
    u_p_t = data[:, 3:4, t_id]
    v_p_t = data[:, 4:5, t_id]
    p_p_t = data[:, 5:6, t_id]
    T_p_t = data[:, 6:7, t_id]

    ax1 = plt.subplot(1, 4, 1)
    print('anim_update()-u: ', t_id, x1_t.shape, x2_t.shape, u_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), u_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), u_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(u_min, u_max):
        plt.clim(vmin=u_min, vmax=u_max * 1.0001)
    cb1 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("u-field at t = " + str(t_id * dt) + " s.", fontsize = 9.5)

    ax2 = plt.subplot(1, 4, 2)
    print('anim_update()-v: ', t_id, x1_t.shape, x2_t.shape, v_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), v_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), v_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(v_min, v_max):
        plt.clim(vmin=v_min, vmax=v_max * 1.0001)
    cb2 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("v-field at t = " + str(t_id * dt) + " s.", fontsize = 9.5)

    ax3 = plt.subplot(1, 4, 3)
    print('anim_update()-p: ', t_id, x1_t.shape, x2_t.shape, p_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), p_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), p_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(p_min, p_max):
        plt.clim(vmin=p_min, vmax=p_max * 1.0001)
    cb3 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("p-field at t = " + str(t_id * dt) + " s.", fontsize = 9.5)

    ax4 = plt.subplot(1, 4, 4)
    print('anim_update()-T: ', t_id, x1_t.shape, x2_t.shape, T_p_t.shape)
    # plt.tricontourf(x1_t.flatten(), x2_t.flatten(), T_p_t.flatten(), cmap="rainbow")
    plt.tricontourf(x1_t.flatten(), x2_t.flatten(), T_p_t.flatten(), levels=30, cmap="rainbow")
    if not np.isclose(T_min, T_max):
        plt.clim(vmin=T_min, vmax=T_max * 1.0001)
    cb4 = plt.colorbar()
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 9.5
    plt.axis("equal")
    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("T-field at t = " + str(t_id * dt) + " s.", fontsize = 9.5)

    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    # fig.tight_layout(pad=0, h_pad=.1, w_pad=.1)
    fig.tight_layout()
    # plt.show()
    plt.savefig(work_path + '//' + f_prefix + '_animation_' + str(t_id) + '.jpg')

# anim_update(10)
# init()

if isTransient:
    # anim = FuncAnimation(fig, anim_update, frames=np.arange(0, data.shape[2]).astype(np.int64), interval=200, init_func=init)
    anim = FuncAnimation(fig, anim_update, frames=np.arange(0, data.shape[2]).astype(np.int64), interval=200)
    anim.save(work_path + "//" + "NC2d-" + time_label + "_animation-" + str(Nt+1) + ".gif", writer="pillow", dpi=300)
else:
    anim_update(0)