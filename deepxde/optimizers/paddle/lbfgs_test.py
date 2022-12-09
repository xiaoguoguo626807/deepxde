from lbfgs_torchoptimizer import LBFGS
import torch  
import paddle 
import numpy as np 

'''
np.random.seed(0)
np_w = np.random.rand(1).astype(np.float32)
np_x = np.random.rand(1).astype(np.float32)

inputs = [np.random.rand(1) for i in range(10)]
# y = 2x
targets = [2 * x for x in inputs]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = torch.nn.Parameter(torch.tensor(np_w))
    
    def forward(self, x):
        return self.w * x 

net = Net()
opt = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')
def train_step(inputs, targets):
    def closure():
        outputs = net(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        print('loss: ', loss.item())
        opt.zero_grad()
        loss.backward()
        return loss 
    opt.step(closure)


for input, target in zip(inputs, targets):
    input = torch.tensor(input)
    target = torch.tensor(target)
    train_step(input, target)

'''
np.random.seed(0)
np_w = np.random.rand(1).astype(np.float32)
np_x = np.random.rand(1).astype(np.float32)

inputs = [np.random.rand(1).astype(np.float32) for i in range(10)]
# y = 2x
targets = [2 * x for x in inputs]

class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        w = paddle.to_tensor(np_w)
        self.w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))
    
    def forward(self, x):
        return self.w * x 

net = Net()
opt = LBFGS(net.parameters(), lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')
def train_step(inputs, targets):
    def closure():
        outputs = net(inputs)
        loss = paddle.nn.functional.mse_loss(outputs, targets)
        print('loss: ', loss.item())
        opt.zero_grad()
        loss.backward()
        return loss 
    opt.step(closure)


for input, target in zip(inputs, targets):
    input = paddle.to_tensor(input)
    target = paddle.to_tensor(target)
    train_step(input, target)
