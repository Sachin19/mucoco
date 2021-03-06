""" Optimizers class """
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import operator
import functools
from copy import copy
from math import sqrt
import types
import importlib
import math
import numpy as np
# from onmt.utils.misc import fn_args

import logging

logger = logging.getLogger(__name__)


def build_torch_optimizer(model, opt):
    """Builds the PyTorch optimizer.

    Args:
      model: The model to optimize.
      opt. The dictionary of options.

    Returns:
      A ``torch.optim.Optimizer`` instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    betas = eval(opt.adam_betas)
    if opt.optim == "sgd":
        optimizer = optim.SGD(list(params), lr=opt.lr, momentum=opt.sgd_momentum, nesterov=opt.sgd_nesterov, weight_decay=opt.weight_decay)
    elif opt.optim == "expgd":
        optimizer = ExpGD(list(params), lr=opt.lr, mw=opt.expgd_mw)
    elif opt.optim == "sgld":
        optimizer = SGLD(list(params), lr=opt.lr, num_burn_in_steps=50)
    elif opt.optim == "lbfgs":
        optimizer = optim.LBFGS(list(params), lr=opt.lr)
    elif opt.optim == "ascentsgd":
        optimizer = optim.SGD(list(params), lr=opt.lambda_lr)
    elif opt.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(list(params), lr=opt.lr)
    elif opt.optim == "adagrad":
        optimizer = optim.Adagrad(
            params,
            lr=opt.lr,
            initial_accumulator_value=opt.adagrad_accumulator_init,
        )
    elif opt.optim == "adadelta":
        optimizer = optim.Adadelta(params, lr=opt.lr)
    elif opt.optim == "adafactor":
        optimizer = AdaFactor(
            params, non_constant_decay=True, enable_factorization=True, weight_decay=0
        )
    elif opt.optim == "adam":
        optimizer = optim.AdamW(
            params,
            lr=opt.lr,
            betas=betas,
            eps=1e-9,
            weight_decay=opt.weight_decay,
        )
    elif opt.optim == "radam":
        optimizer = RAdam(
            params,
            lr=opt.lr,
            betas=betas,
            eps=1e-9,
            weight_decay=opt.weight_decay,
        )
    elif opt.optim == "sparseadam":
        dense = []
        sparse = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # TODO: Find a better way to check for sparse gradients.
            if "embed" in name:
                sparse.append(param)
            else:
                dense.append(param)
        optimizer = MultipleOptimizer(
            [
                optim.Adam(dense, lr=opt.lr, betas=betas, eps=1e-8),
                optim.SparseAdam(sparse, lr=opt.lr, betas=betas, eps=1e-8),
            ]
        )
    elif opt.optim == "fusedadam":
        # we use here a FusedAdam() copy of an old Apex repo
        optimizer = FusedAdam(params, lr=opt.lr, betas=betas)
    else:
        raise ValueError("Invalid optimizer type: " + opt.optim)

    if opt.model_dtype == "fp16" and opt.optim != "ascentsgd" and opt.fp16_source == "apex":
        import apex

        if opt.optim != "fusedadam":
            # In this case use the new AMP API from apex
            loss_scale = "dynamic" #if opt.loss_scale == 0 else opt.loss_scale
            model, optimizer = apex.amp.initialize(
                model,
                optimizer,
                opt_level="O1",
                loss_scale=loss_scale,
                keep_batchnorm_fp32=None,
            )
        else:
            # In this case use the old FusedAdam with FP16_optimizer wrapper
            static_loss_scale = opt.loss_scale
            dynamic_loss_scale = opt.loss_scale == 0
            optimizer = apex.optimizers.FP16_Optimizer(
                optimizer,
                static_loss_scale=static_loss_scale,
                dynamic_loss_scale=dynamic_loss_scale,
            )
    return optimizer


def make_learning_rate_decay_fn(opt):
    """Returns the learning decay function from options."""
    if opt.decay_method is None:
        return None
    if opt.decay_method == "noam":
        return functools.partial(
            noam_decay, warmup_steps=opt.warmup_steps, model_size=opt.rnn_size
        )
    elif opt.decay_method == "noamwd":
        return functools.partial(
            noamwd_decay,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size,
            rate=opt.lr_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps,
        )
    elif opt.decay_method == "rsqrt":
        return functools.partial(rsqrt_decay, warmup_steps=opt.warmup_steps)
    elif opt.decay_method == "linear":
        return functools.partial(
            linear_decay,
            warmup_steps=opt.warmup_steps,
            warmup_end_lr=opt.warmup_end_lr,
            warmup_init_lr=opt.warmup_init_lr,
            max_update=opt.optim_steps,
            min_lr=opt.min_lr,
        )
    elif opt.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=opt.lr_decay,
            decay_steps=opt.decay_steps,
            start_step=opt.start_decay_steps,
        )


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def noamwd_decay(step, warmup_steps, model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches"""
    return (
        model_size ** (-0.5)
        * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        * rate ** (max(step - start_step + decay_steps, 0) // decay_steps)
    )


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def linear_decay(step, min_lr, warmup_steps, warmup_init_lr, warmup_end_lr, max_update):
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
    if step < warmup_steps:
        return warmup_init_lr + step * lr_step
    else:
        return (warmup_end_lr - min_lr) * (
            1 - (step - warmup_steps) / (max_update - warmup_steps)
        ) + min_lr


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def zero_grad(self, set_to_none=False):
        """ ? """
        for op in self.optimizers:
            op.zero_grad(set_to_none=set_to_none)

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.
    """

    def __init__(
        self,
        optimizer,
        learning_rate,
        learning_rate_decay_fn=None,
        decay_method=None,
        max_grad_norm=None,
        ascent=False
    ):
        """Initializes the controller.

        Args:
          optimizer: A ``torch.optim.Optimizer`` instance.
          learning_rate: The initial learning rate.
          learning_rate_decay_fn: An optional callable taking the current step
            as argument and return a learning rate scaling factor.
          max_grad_norm: Clip gradients to this global norm.
        """
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1
        self._fp16 = None
        self.ascent = ascent

        # print(self._optimizer)
        # input()

    @classmethod
    def from_opt(cls, model, opt, checkpoint=None):
        """Builds the optimizer from options.

        Args:
          cls: The ``Optimizer`` class to instantiate.
          model: The model to optimize.
          opt: The dict of user options.
          checkpoint: An optional checkpoint to load states from.

        Returns:
          An ``Optimizer`` instance.
        """
        optim_opt = opt
        optimizer = cls(
            build_torch_optimizer(model, optim_opt),
            optim_opt.lr,
            learning_rate_decay_fn=make_learning_rate_decay_fn(optim_opt),
            decay_method=opt.decay_method,
            max_grad_norm = optim_opt.max_grad_norm,
            ascent=optim_opt.optim == "ascentsgd"
        )
        if opt.model_dtype == "fp16":
            if opt.optim == "fusedadam":
                optimizer._fp16 = "legacy"
            else:
                optimizer._fp16 = "amp"

        return optimizer

    @property
    def training_step(self):
        """The current training step."""
        return self._training_step

    def learning_rate(self, no_improvement=False):
        """Returns the current learning rate."""
        if self._learning_rate_decay_fn is None:
            if no_improvement:
                self._learning_rate /= 2  #half the learning rate if the loss stops decreasing
                logger.info(f"Step {self._decay_step}: Halving the learning rate to {self._learning_rate}")
            if self.ascent:
                return -self._learning_rate 
            return self._learning_rate
        scale = self._learning_rate_decay_fn(self._decay_step)
        # if (self._decay_step % 100 == 0):
        #     print('scale =',scale)
        if self.ascent:
            return -scale * self._learning_rate
        return scale * self._learning_rate

    def state_dict(self):
        return {
            "training_step": self._training_step,
            "decay_step": self._decay_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._training_step = state_dict["training_step"]
        # State can be partially restored.
        if "decay_step" in state_dict:
            self._decay_step = state_dict["decay_step"]
        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])

    def zero_grad(self, set_to_none=False):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss, retain_graph=False, scaler=None):
        """Wrapper for backward pass. Some optimizer requires ownership of the
        backward pass."""
        if scaler is not None:
            scaler.scale(loss).backward(retain_graph=retain_graph)    
        elif self._fp16 == "amp":
            import apex

            with apex.amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self._fp16 == "legacy":
            kwargs = {}
            if "update_master_grads" in fn_args(self._optimizer.backward):
                kwargs["update_master_grads"] = True
            self._optimizer.backward(loss, **kwargs)
        else:
            loss.backward(retain_graph=retain_graph)

    def step(self, no_improvement=False, scaler=None):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        learning_rate = self.learning_rate(no_improvement)

        if self._fp16 == "legacy":
            if hasattr(self._optimizer, "update_master_grads"):
                self._optimizer.update_master_grads()
            if (
                hasattr(self._optimizer, "clip_master_grads")
                and self._max_grad_norm > 0
            ):
                self._optimizer.clip_master_grads(self._max_grad_norm)

        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if scaler is not None and self._max_grad_norm > 0 and not self.ascent:
                scaler.unscale_(self._optimizer)
                clip_grad_norm_(group["params"], self._max_grad_norm)
            elif self._fp16 is None and self._max_grad_norm > 0 and not self.ascent:
                clip_grad_norm_(group["params"], self._max_grad_norm)

            # for p in group['params']:
            #     param_norm = p.grad.data.norm(2, -1).sum(dim=0)
            #     print(param_norm)
            # input()
        
        # for group in self._optimizer.param_groups:
        #     print(group["lr"])
        if scaler is None:
            self._optimizer.step()
        else:
            scaler.step(self._optimizer)
            scaler.update()

        self._decay_step += 1
        self._training_step += 1


# Code below is an implementation of https://arxiv.org/pdf/1804.04235.pdf
# inspired but modified from https://github.com/DeadAt0m/adafactor-pytorch


class AdaFactor(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=None,
        beta1=0.9,
        beta2=0.999,
        eps1=1e-30,
        eps2=1e-3,
        cliping_threshold=1,
        non_constant_decay=True,
        enable_factorization=True,
        ams_grad=True,
        weight_decay=0,
    ):

        enable_momentum = beta1 != 0

        if non_constant_decay:
            ams_grad = False

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps1=eps1,
            eps2=eps2,
            cliping_threshold=cliping_threshold,
            weight_decay=weight_decay,
            ams_grad=ams_grad,
            enable_factorization=enable_factorization,
            enable_momentum=enable_momentum,
            non_constant_decay=non_constant_decay,
        )

        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)

    def _experimental_reshape(self, shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0], shape[1] * shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2
            new_shape = (
                shape[0] * functools.reduce(operator.mul, temp_shape[tmp_div:], 1),
                shape[1] * functools.reduce(operator.mul, temp_shape[:tmp_div], 1),
            )
        return new_shape, copy(shape)

    def _check_shape(self, shape):
        """
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        """
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False

    def _rms(self, x):
        return sqrt(torch.mean(x.pow(2)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse \
                                       gradients, use SparseAdam instead"
                    )

                is_matrix, is_need_reshape = self._check_shape(grad.size())
                new_shape = p.data.size()
                if is_need_reshape and group["enable_factorization"]:
                    new_shape, old_shape = self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if group["enable_momentum"]:
                        state["exp_avg"] = torch.zeros(
                            new_shape, dtype=torch.float32, device=p.grad.device
                        )

                    if is_matrix and group["enable_factorization"]:
                        state["exp_avg_sq_R"] = torch.zeros(
                            (1, new_shape[1]), dtype=torch.float32, device=p.grad.device
                        )
                        state["exp_avg_sq_C"] = torch.zeros(
                            (new_shape[0], 1), dtype=torch.float32, device=p.grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros(
                            new_shape, dtype=torch.float32, device=p.grad.device
                        )
                    if group["ams_grad"]:
                        state["exp_avg_sq_hat"] = torch.zeros(
                            new_shape, dtype=torch.float32, device=p.grad.device
                        )

                if group["enable_momentum"]:
                    exp_avg = state["exp_avg"]

                if is_matrix and group["enable_factorization"]:
                    exp_avg_sq_r = state["exp_avg_sq_R"]
                    exp_avg_sq_c = state["exp_avg_sq_C"]
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                if group["ams_grad"]:
                    exp_avg_sq_hat = state["exp_avg_sq_hat"]

                state["step"] += 1
                lr_t = group["lr"]
                lr_t *= max(group["eps2"], self._rms(p.data))

                if group["enable_momentum"]:
                    if group["non_constant_decay"]:
                        beta1_t = (
                            group["beta1"]
                            * (1 - group["beta1"] ** (state["step"] - 1))
                            / (1 - group["beta1"] ** state["step"])
                        )
                    else:
                        beta1_t = group["beta1"]
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)

                if group["non_constant_decay"]:
                    beta2_t = (
                        group["beta2"]
                        * (1 - group["beta2"] ** (state["step"] - 1))
                        / (1 - group["beta2"] ** state["step"])
                    )
                else:
                    beta2_t = group["beta2"]

                if is_matrix and group["enable_factorization"]:
                    exp_avg_sq_r.mul_(beta2_t).add_(
                        1 - beta2_t,
                        torch.sum(
                            torch.mul(grad, grad).add_(group["eps1"]),
                            dim=0,
                            keepdim=True,
                        ),
                    )
                    exp_avg_sq_c.mul_(beta2_t).add_(
                        1 - beta2_t,
                        torch.sum(
                            torch.mul(grad, grad).add_(group["eps1"]),
                            dim=1,
                            keepdim=True,
                        ),
                    )
                    v = torch.mul(exp_avg_sq_c, exp_avg_sq_r).div_(
                        torch.sum(exp_avg_sq_r)
                    )
                else:
                    exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_(
                        (1 - beta2_t) * group["eps1"]
                    )
                    v = exp_avg_sq

                g = grad
                if group["enable_momentum"]:
                    g = torch.div(exp_avg, 1 - beta1_t ** state["step"])

                if group["ams_grad"]:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat
                    u = torch.div(
                        g,
                        (torch.div(v, 1 - beta2_t ** state["step"]))
                        .sqrt()
                        .add_(group["eps1"]),
                    )
                else:
                    u = torch.div(g, v.sqrt())

                u.div_(max(1, self._rms(u) / group["cliping_threshold"]))
                p.data.add_(
                    -lr_t
                    * (
                        u.view(old_shape)
                        if is_need_reshape and group["enable_factorization"]
                        else u
                    )
                )

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"] * lr_t, p.data)

        return loss


class FusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.
       Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square.
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        max_grad_norm=0.0,
        amsgrad=False,
    ):
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError("AMSGrad variant not supported.")
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def step(
        self, closure=None, grads=None, output_params=None, scale=1.0, grad_norms=None
    ):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients.
                (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        for group, grads_this_group, output_params_this_group, grad_norm in zip(
            self.param_groups, grads_group, output_params_group, grad_norms
        ):
            if grads_this_group is None:
                grads_this_group = [None] * len(group["params"])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group["params"])

            # compute combined scale factor for this group
            combined_scale = scale
            if group["max_grad_norm"] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group["max_grad_norm"]
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group["bias_correction"] else 0

            for p, grad, output_param in zip(
                group["params"], grads_this_group, output_params_this_group
            ):
                # note: p.grad should not ever be set for correct operation of
                # mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse \
                                       gradients, please consider \
                                       SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = (
                    torch.tensor([], dtype=torch.float)
                    if output_param is None
                    else output_param
                )
                fused_adam_cuda.adam(
                    p.data,
                    out_p,
                    exp_avg,
                    exp_avg_sq,
                    grad,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    combined_scale,
                    state["step"],
                    self.eps_mode,
                    bias_correction,
                    group["weight_decay"],
                )
        return loss


class RAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    step_size = (
                        group["lr"]
                        * math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                        / (1 - beta1 ** state["step"])
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    step_size = group["lr"] / (1 - beta1 ** state["step"])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["warmup"] > state["step"]:
                    scheduled_lr = 1e-8 + state["step"] * group["lr"] / group["warmup"]
                else:
                    scheduled_lr = group["lr"]

                step_size = (
                    scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                )

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

class SGLD(torch.optim.Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """
    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.99,
                 num_pseudo_batches=1,
                 num_burn_in_steps=3000,
                 diagonal_bias=1e-8) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                #  }}} State initialization #

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) * ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                    1. / torch.sqrt(momentum + group["diagonal_bias"])
                )

                scaled_grad = (
                    0.5 * preconditioner * gradient * num_pseudo_batches +
                    torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma * torch.sqrt(preconditioner)
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss

class ExpGD(torch.optim.Optimizer):
    """ Exponentiated Gradient Descent
    """
    def __init__(self,
                 params,
                 lr=1e-2,
                 mw=1, momentum=0.9) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            mw=mw,
            momentum=momentum,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(ExpGD, self).__setstate__(state)

    def step(self):
        loss = None

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                gradient = parameter.grad.data

                mw = group["mw"]
                momentum = group['momentum']

                # print(parameter.size())
                # print(gradient.size())
                
                # print(parameter)
                if mw == 1:
                    unnormalized = parameter.data.mul_(torch.exp(-lr * gradient))
                    # print(unnormalized)
                    parameter.data.div_(unnormalized.sum(dim=-1, keepdims=True))
                    # print(-lr * gradient)
                    # print(torch.exp(-lr * gradient))
                    # print(parameter)
                    # print(parameter.max(-1))
                    # input()
                elif mw == 3: # variation
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["step"] = 0
                        state["mt"] = torch.zeros_like(gradient)
                    mt = state['mt']
                    step = state['step']
                    exponand = -lr * gradient - 4 * lr * lr * (gradient - mt) ** 2
                    unnormalized = parameter.data.mul_(torch.exp(exponand))
                    # print(unnormalized)
                    parameter.data.div_(unnormalized.sum(dim=-1, keepdims=True))

                    mt = mt * step + gradient 
                    mt = mt / (step + 1)
                    state['step'] += 1
                    state['mt'] = mt
                elif mw == 4: #optimistic
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["step"] = 0
                        state["zt"] = torch.zeros_like(gradient)
                    zt = state['zt']
                    step = state['step']

                    exponand = -lr * gradient - lr * lr * (gradient - zt) ** 2
                    unnormalized = parameter.data.mul_(torch.exp(exponand))
                    # print(unnormalized)
                    parameter.data.div_(unnormalized.sum(dim=-1, keepdims=True))

                    state['step'] += 1
                    state['zt'] = gradient
                elif mw == 5: #momentum
                    state = self.state[parameter]
                    if len(state) == 0:
                        state['step'] = 0
                        state['lastgrad'] = torch.zeros_like(gradient)
                    lastgrad = state['lastgrad']
                    step = state['step']
                    grad = (momentum * lastgrad + lr * gradient)
                    unnormalized = parameter.data.mul_(torch.exp(-grad))
                    # print(unnormalized)
                    parameter.data.div_(unnormalized.sum(dim=-1, keepdims=True))

                    state['step'] += 1
                    state['lastgrad'] = grad
                
                elif mw == 6: #momentum with no exponent
                    state = self.state[parameter]
                    if len(state) == 0:
                        state['step'] = 0
                        state['lastgrad'] = torch.zeros_like(gradient)
                    lastgrad = state['lastgrad']
                    step = state['step']
                    grad = (momentum * lastgrad + lr * gradient)
                    unnormalized = parameter.data.mul_(1-gradient)
                    # print(unnormalized)
                    parameter.data.div_(unnormalized.sum(dim=-1, keepdims=True))

                    state['step'] += 1
                    state['lastgrad'] = grad

                else:
                    unnormalized = parameter.data.mul_(1-lr * gradient)
                    # print(unnormalized)
                    parameter.data.div_(unnormalized.sum(dim=-1, keepdims=True))
                

                # print(parameter.size())
                # print(parameter.data.sum(dim=-1))
                # input()

        return loss
