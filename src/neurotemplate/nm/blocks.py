"""
Block, Linear, MLP, MLP_bounds, and the two bound functions MLP_bounds uses.

From neuromancer/modules/blocks.py and neuromancer/modules/functions.py.
Removed: every other block, `reg_error`, and the `linear_map`/`nonlin`
defaults (`slim.Linear`, `SoftExponential`), which NeuroTemplate always
overrides; here they are required keywords.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def bounds_scaling(x, xmin, xmax, scaling=1.):
    """
    hard bounds on variable x via sigmoid scaling between xmin and xmax values
    :param x:
    :param xmin:
    :param xmax:
    :param scaling:
    :return:
    """
    x = (xmax - xmin) * torch.sigmoid(scaling * x) + xmin
    return x


def bounds_clamp(x, xmin=None, xmax=None):
    """
    hard bounds on variable x via ReLU clamping between xmin and xmax values
    :param x:
    :param xmin:
    :param xmax:
    :return:
    """
    if xmin is not None:
        x = x + torch.relu(-x + xmin)
    if xmax is not None:
        x = x - torch.relu(x - xmax)
    return x


class Block(nn.Module, ABC):
    """
    Canonical abstract class of the block function approximator
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def block_eval(self, x):
        pass

    def forward(self, *inputs):
        """
        Handling varying number of tensor inputs

        :param inputs: (list(torch.Tensor, shape=[batchsize, insize]) or torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        return self.block_eval(x)


class Linear(Block):
    """
    Linear map consistent with block interface.
    """

    def __init__(
        self,
        insize,
        outsize,
        *,
        linear_map,
        bias=True,
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param linear_map: (class) Linear map class, e.g. torch.nn.Linear
        :param bias: (bool) Whether to use bias
        :param linargs: (dict) Arguments for instantiating linear layer
        """

        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.linear = linear_map(insize, outsize, bias=bias, **linargs)

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        return self.linear(x)


class MLP(Block):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """

    def __init__(
        self,
        insize,
        outsize,
        *,
        linear_map,
        nonlin,
        hsizes,
        bias=True,
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param linear_map: (class) Linear map class, e.g. torch.nn.Linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param bias: (bool) Whether to use bias
        :param linargs: (dict) Arguments for instantiating linear layer
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList(
            [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
        )
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x


class MLP_bounds(MLP):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """

    bound_methods = {"sigmoid_scale": bounds_scaling, "relu_clamp": bounds_clamp}

    def __init__(
        self,
        insize,
        outsize,
        *,
        linear_map,
        nonlin,
        hsizes,
        bias=True,
        linargs=dict(),
        min=0.0,
        max=1.0,
        method="sigmoid_scale",
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param linear_map: (class) Linear map class, e.g. torch.nn.Linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param bias: (bool) Whether to use bias
        :param linargs: (dict) Arguments for instantiating linear layer
        :param min: lower bound of the output
        :param max: upper bound of the output
        :param method: a key of bound_methods, or a differentiable callable
        """

        super().__init__(
            insize=insize,
            outsize=outsize,
            bias=bias,
            linear_map=linear_map,
            nonlin=nonlin,
            hsizes=hsizes,
            linargs=linargs,
        )

        self.min = min
        self.max = max
        self.method = self._set_method(method)

    def _set_method(self, method):
        if method in self.bound_methods.keys():
            return self.bound_methods[method]
        else:
            assert callable(method), (
                f"Method, {method} must be a key in {self.bound_methods} "
                f"or a differentiable callable."
            )
            return method

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return self.method(x, self.min, self.max)
