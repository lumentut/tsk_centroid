from __future__ import annotations

from pyit2fls import IT2TSK
from .fuzzy_logic import InputVariable, OutputVariable, Rule
from . import BaseFIS
from src.utils.norm_mapper import s_norm_fn, t_norm_fn


class IT2TskFIS(BaseFIS):
    def __init__(
        self,
        input_variables: list[InputVariable] = [],
        output_variables: list[OutputVariable] = [],
        rules: list[Rule] = [],
        t_norm: str = "min_t_norm",
        s_norm: str = "max_s_norm",
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.rules = rules
        self.t_norm = t_norm
        self.s_norm = s_norm
        self.fis = None

    def create_fis(self) -> IT2TskFIS:
        self.fis = IT2TSK(t_norm=t_norm_fn(self.t_norm), s_norm=s_norm_fn(self.s_norm))
        return self

    def add_input_variables(self) -> IT2TskFIS:
        for var in self.input_variables:
            self.fis.add_input_variable(var["name"])
        return self

    def add_output_variables(self) -> IT2TskFIS:
        for var in self.output_variables:
            self.fis.add_output_variable(var["name"])
        return self

    def add_rules(self) -> IT2TskFIS:
        for rule in self.rules:
            self.fis.add_rule(rule["antecedents"], rule["consequents"])
        return self

    def build(self) -> IT2TSK:
        return (
            self.create_fis()
            .add_input_variables()
            .add_output_variables()
            .add_rules()
            .fis
        )
