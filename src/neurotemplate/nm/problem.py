"""
Problem: the nodes, then the loss, as one module.

From neuromancer/problem.py. Removed: LitProblem (Lightning) and graph
drawing. `_resolve_graph` keeps what upstream's `graph()` did besides
drawing: naming unnamed nodes and checking every name is unique.
"""
import warnings
from typing import Callable, Dict, List

import torch
import torch.nn as nn


class Problem(nn.Module):
    """
    This class is similar in spirit to a nn.Sequential module. However,
    by concatenating input and output dictionaries for each node
    module we can represent arbitrary directed acyclic computation graphs.
    In addition the Problem module takes care of calculating loss functions
    via given instantiated weighted multi-objective PenaltyLoss object which
    calculate objective and constraints terms from aggregated input and set
    of outputs from the node modules.
    """

    def __init__(self, nodes: List[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]],
                 loss: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
                 grad_inference=False, check_overwrite=False):
        """
        :param nodes: (List[Node]) list of objects which implement the Node interface
                                   (i.e. input and output are dicts of Tensors and
                                   object has input_keys, output_keys, and name attributes)
        :param loss: (PenaltyLoss) instantiated loss class
        :param grad_inference: (boolean) flag for enabling computation of grdients during inference time, useful for techniques like projected gradient
        """
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.loss = loss
        self.grad_inference = grad_inference
        self.check_overwrite = check_overwrite
        self._check_keys()
        self._resolve_graph()

    def _check_keys(self):
        keys = set()
        for node in list(self.nodes)+[self.loss]:
            keys |= set(node.input_keys)
            new_keys = set(node.output_keys)
            same = new_keys & keys
            if self.check_overwrite:
                if len(same) != 0:
                    warnings.warn(f'Keys {same} are being overwritten by the node {node}.')
            keys |= new_keys

    def _check_unique_names(self):
        num_unique = len(set([o.name for o in self.loss.objectives] + [c.name for c in self.loss.constraints]
                             + [comp.name for comp in self.nodes]))
        num_obj = len(self.loss.objectives) + len(self.loss.constraints) + len(self.nodes)
        assert num_unique == num_obj, \
            "All nodes, objectives and constraints must have unique names to construct a computational graph."

    def _resolve_graph(self):
        """Upstream's `graph()` without the drawing."""
        self._check_unique_names()
        input_keys = []
        output_keys = []
        nonames = 1
        for node in self.nodes:
            input_keys += node.input_keys
            output_keys += node.output_keys
            if node.name is None or node.name == '':
                node.name = f'node_{nonames}'
                nonames += 1
        input_keys += self.loss.input_keys
        self.input_keys = list(set(input_keys))
        output_keys += self.loss.output_keys
        self.output_keys = list(set(output_keys))

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.step(data)
        output_dict = self.loss(output_dict)
        if isinstance(output_dict, torch.Tensor):
            output_dict = {self.loss.name: output_dict}
        return {f'{data["name"]}_{k}': v for k, v in output_dict.items()}

    def step(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for node in self.nodes:
            output_dict = node(input_dict)
            if isinstance(output_dict, torch.Tensor):
                output_dict = {node.name: output_dict}
            input_dict = {**input_dict, **output_dict}
        return input_dict

    def predict(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            output_dict = self.step(data)
        return {k: v for k,v in output_dict.items()}

    def freeze(self):
        """
        Freezes the parameters of all nodes in the system
        """
        for node in self.nodes:
            node.freeze()

    def unfreeze(self):
        """
        Unfreezes the parameters of all nodes in the system
        """
        for node in self.nodes:
            node.unfreeze()

    def __repr__(self):
        s = "### MODEL SUMMARY ###\n\nNODES:"
        if len(self.nodes) > 0:
            for c in self.nodes:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        s += "\nCONSTRAINTS:"
        if len(self.loss.constraints) > 0:
            for c in self.loss.constraints:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        s += "\nOBJECTIVES:"
        if len(self.loss.objectives) > 0:
            for c in self.loss.objectives:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        return s
