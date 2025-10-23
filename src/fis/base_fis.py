from abc import ABC, abstractmethod
from .fuzzy_logic import InputVariable, OutputVariable, Rule


class BaseFIS(ABC):
    """Abstract base class for Fuzzy Inference Systems (FIS) builder."""

    @abstractmethod
    def create_fis(self):
        pass

    @abstractmethod
    def add_input_variables(self, input_variables: list[InputVariable]):
        pass

    @abstractmethod
    def add_output_variables(self, output_variables: list[OutputVariable]):
        pass

    @abstractmethod
    def add_rules(self, rules: list[Rule]):
        pass

    @abstractmethod
    def build(self):
        pass
