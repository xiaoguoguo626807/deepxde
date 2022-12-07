import paddle 
import warnings
from functools import reduce 
from collections import defaultdict
from paddle.optimizer import Optimizer

__all__ = ['Optimizer']

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    CUBIC_LOG=False
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    print(" x1, xmin, f1, g1 = ", x1, xmin_bound, f1, g1)if CUBIC_LOG else None
    print(" x2, xmax, f2, g2 = ", x2, xmax_bound, f2, g2)if CUBIC_LOG else None
    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    print("g1 + g2 = ", g1 + g2)if CUBIC_LOG else None
    print("f1 - f2 = ", f1 - f2)if CUBIC_LOG else None
    print("x1 - x2 = ", x1 - x2)if CUBIC_LOG else None
    print("3 * (f1 - f2)", 3 * (f1 - f2))if CUBIC_LOG else None
    print("- 3 * (f1 - f2)",- 3 * (f1 - f2))if CUBIC_LOG else None
    print("g1 + g2 - 3 * (f1 - f2) = ", g1 + g2 - 3 * (f1 - f2))if CUBIC_LOG else None

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    print("d1 = ", d1)if CUBIC_LOG else None
    print("d2_square = ", d2_square)if CUBIC_LOG else None
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        print("d2 = ", d2)if CUBIC_LOG else None
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            print("min_pos1 = ",min_pos)if CUBIC_LOG else None
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            print("min_pos2 = ",min_pos)if CUBIC_LOG else None
        print("min_pos = ",min_pos)if CUBIC_LOG else None
        print("cubic return alpha 1= ", min(max(min_pos, xmin_bound), xmax_bound))if CUBIC_LOG else None
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        print("cubic return alpha 2= ",(xmin_bound + xmax_bound) / 2.)if CUBIC_LOG else None
        return (xmin_bound + xmax_bound) / 2.

STRONG_LOG=False
def _strong_wolfe(obj_func,
                  x,
                  alpha,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    print("d_norm = ", d_norm)
    g = g.clone()
    # evaluate objective and gradient using initial step
    print("xk = ", x) if STRONG_LOG else None
    print("alpha = ", alpha) if STRONG_LOG else None
    print("pk = ", d) if STRONG_LOG else None
    f_new, g_new = obj_func(x, alpha, d)
    print("strong wholf First func value : ", f_new) if STRONG_LOG else None
    print("strong wholf First func grad : ", g_new) if STRONG_LOG else None
    print("strong wholf First func pk : ", d) if STRONG_LOG else None
    ls_func_evals = 1
    gtd_new = paddle.dot(g_new, d)

    print("strong wholf First func gtd : ", gtd_new) if STRONG_LOG else None
    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = paddle.to_tensor(0, dtype=g.dtype), f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        print("[[[[strong_wholf first while]]]] : ls_iter :", ls_iter)
        # check conditions
        print("loss_new = ",f_new)
        print( "(f + c1 * alpha * gtd) = ",(f + c1 * alpha * gtd).numpy())
        print(" gtd_new = ", gtd_new.numpy())
        print(" (-c2*gtd) = ", (-c2*gtd))
        if f_new > (f + c1 * alpha * gtd) or (ls_iter > 1 and f_new >= f_prev):
            if f_new > (f + c1 * alpha * gtd):
                print("@@ 1 @@ f+c1*alpha*gtd = ",f + c1 * alpha * gtd, "f_new = ",f_new)
            if ls_iter > 1 and f_new >= f_prev:
                print("@@ 1 @@ f_new =", f_new, "f_prev = ", f_prev)
            bracket = [t_prev, alpha]
            bracket_f = [f_prev, f_new]
            # bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_g = [g_prev, g_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if paddle.abs(gtd_new) <= -c2 * gtd:
            print("@@ 2 @@ paddle.abs(gtd_new) <= -c2 * gtd: gtd_new = ", gtd_new,  "-c2*gtd = ", -c2*gtd)
            bracket = [alpha]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            print("@@ 3@@@ gtd_new >= 0")
            bracket = [t_prev, alpha]
            bracket_f = [f_prev, f_new]
            # bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_g = [g_prev, g_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        print("@@@@@strong_wholf first while interpolate")
        min_step = alpha + 0.01 * (alpha - t_prev)
        max_step = alpha * 10
        tmp = alpha
        alpha = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            alpha,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))
        print("first while cubic alpha : ", alpha) #if STRONG_LOG else None

        # next step
        t_prev = tmp
        f_prev = f_new
        # g_prev = g_new.clone(memory_format=torch.contiguous_format)
        g_prev = g_new.clone()
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, alpha, d)
        print("strong wholf First while funcloss : ", f_new)# if STRONG_LOG else None
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, alpha]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        print("[[[[strong_wholf second while]]]] : ls_iter :", ls_iter)
        # line-search bracket is so small
        if paddle.abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            print("@@@@@ param update small than tol_change break")
            break

        print("bracket [0] number:",bracket[0], bracket_f[0], bracket_gtd[0].numpy())# if STRONG_LOG else None
        print("bracket [1] number:",bracket[1], bracket_f[1], bracket_gtd[1].numpy())# if STRONG_LOG else None
        # compute new trial value
        alpha = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])
        print("second while cubic alpha : ", alpha) #if STRONG_LOG else None
        # test that we are making sufficient progress:
        # in case `alpha` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `alpha` is at one of the boundary,
        # we will move `alpha` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        print("second while eps : ", eps) if STRONG_LOG else None
        if min(max(bracket) - alpha, alpha - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or alpha >= max(bracket) or alpha <= min(bracket):
                # evaluate at 0.1 away from boundary
                if paddle.abs(alpha - max(bracket)) < paddle.abs(alpha - min(bracket)):
                    alpha = max(bracket) - eps
                    print("alpha = max_alpha - eps : ",alpha)if STRONG_LOG else None
                else:
                    alpha = min(bracket) + eps
                    print("alpha = min_alpha + eps : ",alpha)if STRONG_LOG else None
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False
        print("second while final alpha : ", alpha)# if STRONG_LOG else None
        # Evaluate new point
        f_new, g_new = obj_func(x, alpha, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1
         
        print("loss_new = ",f_new)
        print("(f + c1 * alpha * gtd) = ",(f + c1 * alpha * gtd).cpu().numpy())
        print(" gtd_new = ", gtd_new.cpu().numpy())
        print(" (-c2*gtd) = ", (-c2*gtd))

        if f_new > (f + c1 * alpha * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            if f_new > (f + c1 * alpha * gtd):
                print("second while @@ 1 @@ loss_new > (loss + c1 * alpha * gtd) loss_new: ",f_new, "loss + c1 * alpha * gtd",f + c1 * alpha * gtd)
            if f_new >= bracket_f[low_pos]:
                print("second while @@ 1 @@ loss_new >= bracket_loss[low_pos]loss_new:  ",f_new, "bracket_loss[low_pos]",bracket_f[low_pos])
            bracket[high_pos] = alpha
            bracket_f[high_pos] = f_new
            # bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_g[high_pos] = g_new.clone()
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if paddle.abs(gtd_new) <= -c2 * gtd:
                print("second while @@ 2 @@ paddle.abs(gtd_new) <= -c2 * gtd")
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                print("second while @@ 3 @@ gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0  gtd_new: ",gtd_new, "detla alpha = ",(bracket[high_pos] - bracket[low_pos]))
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = alpha
            bracket_f[low_pos] = f_new
            # bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_g[low_pos] = g_new.clone()
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    alpha = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, alpha, ls_func_evals

LOG=False
class LBFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Args:
        lr (float, optional): learning rate .The default value is 1
        max_iter (int, optional): maximal number of iterations per optimization step.
            The default value is 20.
        max_eval (int, optional): maximal number of function evaluations per optimization
            step. The default value is max_iter * 1.25.
        tolerance_grad (float, optional): termination tolerance on first order optimality
            The default value is 1e-5.
        tolerance_change (float, optional): termination tolerance on function
            value/parameter changes. The default value is 1e-9.
        history_size (int, optional): update history size. The default value is 100.
        line_search_fn (string, optional): either 'strong_wolfe' or None. The default value is strong_wolfe.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of \
            some derived class of ``GradientClipBase`` . There are three cliping strategies \
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , \
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None. 

    Examples:
    .. code-block:: python

    import paddle
    import numpy as np 
    from paddle.incubate.optimizer.functional import LBFGS

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
    opt = LBFGS(lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe', parameters=net.parameters())
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
    """

    def __init__(self,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        # other_params = dict(
        #     lr=lr,
        #     max_iter=max_iter,
        #     max_eval=max_eval,
        #     tolerance_grad=tolerance_grad,
        #     tolerance_change=tolerance_change,
        #     history_size=history_size,
        #     line_search_fn=line_search_fn)
        self.lr = lr
        self.max_iter=max_iter
        self.max_eval=max_eval
        self.tolerance_grad=tolerance_grad
        self.tolerance_change=tolerance_change
        self.history_size=history_size
        self.line_search_fn=line_search_fn
        
        self.other_params = {
            'lr': lr,
            'max_iter': max_iter,
            'max_eval': max_eval,
            'tolerance_grad': tolerance_grad,
            'tolerance_change': tolerance_change,
            'history_size': history_size,
            'line_search_fn': line_search_fn,
        }

        if isinstance(parameters, paddle.Tensor):
            raise TypeError("parameters argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            Paddle.typename(parameters))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(parameters)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        super().__init__(
            learning_rate=1.0,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        
        self._default_dict = self.other_params

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")
        # self._params = self.param_groups[0]['params']

        if not isinstance(self._parameter_list[0], dict):
            self._params = self._parameter_list
        else:
            for idx, param_group in enumerate(self._param_groups):
             self._params = param_group['params']
        
        self._numel_cache = None

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, paddle.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

        
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, paddle.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + param.dtype)
            # if not self.other_params.get('differentiable', None) and not (param.is_leaf or param.retains_grad):
            #     raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.other_params.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        # params = param_group['params']
        # if len(params) != len(set(params)):
        #     warnings.warn("optimizer contains a parameter group with duplicate parameters; "
        #                   "in future, this will cause an error; ", stacklevel=3)

        # param_set = set()
        # for group in self.param_groups:
        #     param_set.update(set(group['params']))

        # if not param_set.isdisjoint(set(param_group['params'])):
        #     raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    # compute the number of all parameters
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    # flatten grad of all parameters 
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                # view = p.new(p.numel()).zero_()
                view = paddle.zeros_like(p).reshape([-1])
            elif p.grad.is_sparse():
                view = p.grad.to_dense().reshape([-1])
            else:
                view = p.grad.reshape([-1])
            views.append(view)
        return paddle.concat(views, axis=0)

    # compute xk = xk + alpha * direction
    def _add_grad(self, alpha, direction):
        offset = 0
        for p in self._params:
            numel = p.numel().item()
            p = paddle.assign(p.add(direction[offset:offset+numel].reshape(p.shape) * alpha), p)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        # return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
        return [p.clone() for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # p.copy_(pdata)
            paddle.assign(pdata, p)

    def _directional_evaluate(self, closure, x, alpha, d):
        self._add_grad(alpha, d)
        params=[]
        for p in self._params:
            params.append(p.reshape([-1]))
        paddle.concat(params, 0)
        print("xk + alpha*pk = ", params) if STRONG_LOG else None
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @paddle.no_grad()
    def step(self, closure):
        """Performs a single optimization step.
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        # closure = torch.enable_grad()(closure)
        closure = paddle.set_grad_enabled(True)(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
        print("[[[[[[[[[[[[[new lbfgs step]]]]]]]]]]]")

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        print("First func value : ", loss)# if LOG else None
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            print("grad is smaller than tolerance_grad, no need to use line search fn")
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        alpha = state.get('alpha')
        old_yk = state.get('old_yk')
        old_sk = state.get('old_sk')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            print("&&&&&&&&&&& n_iter &&&&&&&&&& = ", n_iter)#if LOG else None
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_yk = []
                old_sk = []
                ro = []
                # H_diag = 1
                H_diag = paddle.to_tensor(1., dtype=orig_loss.dtype)
            else:
                # do lbfgs update (update memory)
                # y = flat_grad.sub(prev_flat_grad)
                y = flat_grad.subtract(prev_flat_grad)
                # s = d.mul(alpha)
                s = d.multiply(paddle.to_tensor(alpha, dtype=d.dtype))
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_yk) == history_size:
                        # shift history by one (limited-memory)
                        old_yk.pop(0)
                        old_sk.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_yk.append(y)
                    old_sk.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)
                    print("H_diag : ", H_diag )#if LOG else None
                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_yk)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_sk[i].dot(q) * ro[i]
                    # q.add_(old_yk[i], alpha=-al[i])
                    paddle.assign(q.add(old_yk[i]*(-al[i])), q)

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = paddle.multiply(q, H_diag)
                for i in range(num_old):
                    be_i = old_yk[i].dot(r) * ro[i]
                    # r.add_(old_sk[i], alpha=al[i] - be_i)
                    paddle.assign(r.add(old_sk[i] * (al[i]-be_i)), r)

            print("d = ", d)if LOG else None

            if prev_flat_grad is None:
                # prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
                prev_flat_grad = flat_grad.clone()
            else:
                # prev_flat_grad.copy_(flat_grad)
                paddle.assign(flat_grad, prev_flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                alpha = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                alpha = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d
            print("gtd = ", gtd)# if LOG else None
            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                print("@@@@@ gtd > -tolerance_change")
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, alpha, d):
                        return self._directional_evaluate(closure, x, alpha, d)

                    loss, flat_grad, alpha, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, alpha, d, loss, flat_grad, gtd)
                print("strongwolfe update alpha = ", alpha)
                self._add_grad(alpha, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(alpha, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    # with torch.enable_grad():
                    with paddle.set_grad_enabled(True):
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if (d * alpha).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['alpha'] = alpha
        state['old_yk'] = old_yk
        state['old_sk'] = old_sk
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss


