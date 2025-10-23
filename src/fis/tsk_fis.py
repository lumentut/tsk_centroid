from __future__ import annotations

from pyit2fls import T1TSK
from .fuzzy_logic import InputVariable, OutputVariable, Rule
from . import BaseFIS


class TskFIS(BaseFIS):
    def __init__(
        self,
        input_variables: list[InputVariable] = [],
        output_variables: list[OutputVariable] = [],
        rules: list[Rule] = [],
    ):
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.rules = rules
        self.fis = None

    def create_fis(self) -> TskFIS:
        self.fis = T1TSK()
        return self

    def add_input_variables(self) -> TskFIS:
        for var in self.input_variables:
            self.fis.add_input_variable(var["name"])
        return self

    def add_output_variables(self) -> TskFIS:
        for var in self.output_variables:
            self.fis.add_output_variable(var["name"])
        return self

    def add_rules(self) -> TskFIS:
        for rule in self.rules:
            self.fis.add_rule(rule["antecedents"], rule["consequents"])
        return self

    def build(self) -> T1TSK:
        return (
            self.create_fis()
            .add_input_variables()
            .add_output_variables()
            .add_rules()
            .fis
        )
