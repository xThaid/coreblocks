from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coreblocks.params.genparams import GenParams
    from coreblocks.frontend.bpu.component import BPUComponent

__all__ = [
    "BPUComponentConfig",
    "BPUPredictorConfig",
    "BranchPredictionConfig",
    "MicroBTBConfig",
]


@dataclass(frozen=True)
class BPUComponentConfig(ABC):
    """Configuration of a single sub-component of the branch prediction unit."""

    def meta_width(self, fetch_width: int) -> int:
        """Width of the metadata this component produces with every prediction and needs
        back at training time."""
        return 0

    def validate(self) -> None:
        """Raise ValueError if this component's parameters are incorrect."""


@dataclass(frozen=True)
class BPUPredictorConfig(BPUComponentConfig):
    """Configuration of a component that predicts, i.e. participates in the BPU pipeline."""

    @abstractmethod
    def get_module(self, gen_params: "GenParams") -> "BPUComponent":
        raise NotImplementedError()


@dataclass(frozen=True)
class MicroBTBConfig(BPUPredictorConfig):
    """Configuration of the micro-BTB."""

    entries_log: int = 3
    """Log of the number of entries."""

    useful_cnt_width: int = 2
    """Width of the per-entry saturating usefulness counter that drives replacement."""

    def validate(self):
        if self.entries_log < 1:
            raise ValueError("Micro-BTB must have at least 2 entries")
        if self.useful_cnt_width < 1:
            raise ValueError("Micro-BTB usefulness counter must be at least 1 bit wide")

    def get_module(self, gen_params: "GenParams") -> "BPUComponent":
        from coreblocks.frontend.bpu.micro_btb import MicroBTB

        return MicroBTB(gen_params, self)


@dataclass(frozen=True)
class BranchPredictionConfig:
    """Configuration of the branch prediction unit and all of its sub-predictors."""

    predictors: tuple[BPUPredictorConfig, ...] = (MicroBTBConfig(),)
    """The components of the prediction pipeline, ordered by priority: later components
    override earlier ones. The order also defines the layout of the concatenated
    prediction metadata."""

    def components(self) -> tuple[BPUComponentConfig, ...]:
        return self.predictors

    def bpd_meta_width(self, fetch_width: int) -> int:
        return sum(component.meta_width(fetch_width) for component in self.components())

    def validate(self):
        for component in self.components():
            component.validate()
