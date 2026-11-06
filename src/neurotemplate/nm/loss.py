"""
Loss terms and their penalty-method sum.

`Loss` is from neuromancer/constraint.py; `AggregateLoss` and `PenaltyLoss`
from neuromancer/loss.py. Removed: the symbolic Variable/Constraint algebra,
the barrier and augmented-Lagrangian losses, and loss arithmetic
(`+`, `*`). NeuroTemplate passes no constraints, so `AggregateLoss`
refuses them rather than carry the constraint bookkeeping.
"""
from typing import Callable, Dict, List

import torch
import torch.nn as nn


class Loss(nn.Module):
    """
    Drop in replacement for a Constraint object but relies on a list of dictionary keys and a callable function
    to instantiate.
    """
    def __init__(self, input_keys: List[str], loss: Callable[..., torch.Tensor], weight=1.0, name='loss'):
        """

        :param input_keys: List of str
        :param loss: (callable) Number of arguments of the callable should equal the number of strings in variable names.
                                Arguments to callable should be torch.Tensor and return type a 0-dimensional torch.Tensor
        :param weight: (float) Weight of loss for calculating multi-term loss function
        :param name: (str) Name for tracking output
        """
        super().__init__()
        self.name, self.input_keys, self.output_keys = name, input_keys, [name]
        self.weight = weight
        self.loss = loss

    def forward(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param variables: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :return: 0-dimensional torch.Tensor that can be cast as a floating point number
        """
        return {self.output_keys[0]: self.weight*self.loss(*[variables[k] for k in self.input_keys])}

    def __repr__(self):
        return f"Loss: {self.name}({', '.join(self.input_keys)}) -> {self.loss} * {self.weight}"


class AggregateLoss(nn.Module):
    """
    Abstract aggregate loss class for calculating constraints, objectives, and aggegate loss values.
    """
    def __init__(self, objectives, constraints):
        """
        :param objectives: (list (Loss)) list of objective terms
        :param constraints: (list) must be empty
        """
        super().__init__()
        if constraints:
            raise ValueError("Constraints are not supported; express each one as a Loss term.")
        self.objectives = nn.ModuleList(objectives)
        self.constraints = nn.ModuleList()

        input_keys = []
        for obj in self.objectives:
            input_keys += obj.input_keys
        self.input_keys = list(set(input_keys))
        self.output_keys = ['loss', 'objective_loss', 'penalty_loss',
                            'C_violations', 'C_values', 'C_eq_violations',
                            'C_ineq_violations', 'C_eq_values', 'C_ineq_values']
        self._check_keys()

    def _check_keys(self):
        keys = set()
        for loss in self.objectives:
            keys |= set(loss.input_keys)
            new_keys = set(loss.output_keys)
            same = new_keys & keys
            assert len(same) == 0, \
                f'Keys {same} are being overwritten by the loss term {loss}.'
            keys |= new_keys

    def calculate_objectives(self, input_dict):
        """
        Calculate the value of the objective function for SGD
        """
        loss = 0.0
        output_dict = {}
        for objective in self.objectives:
            output = objective(input_dict)
            if isinstance(output, torch.Tensor):
                output = {objective.output_keys[0]: output}
            output_dict = {**output_dict, **output}
            loss += output_dict[objective.output_keys[0]]
        output_dict['objective_loss'] = loss
        return output_dict


class PenaltyLoss(AggregateLoss):
    """
    Penalty loss function.
        https://en.wikipedia.org/wiki/Penalty_method
    """

    def forward(self, input_dict):
        """

        :param input_dict: (dict {str: torch.Tensor}) Values from forward pass calculations
        :return: (dict {str: torch.Tensor}) input_dict appended with calculated loss values
        """
        objectives_dict = self.calculate_objectives(input_dict)
        input_dict = {**input_dict, **objectives_dict}
        fx = objectives_dict['objective_loss']
        # With no constraints upstream's penalty is the float 0.0.
        penalties = 0.0
        input_dict['penalty_loss'] = penalties
        input_dict['loss'] = fx + penalties
        return input_dict
