from dataclasses import dataclass

__all__ = [
    "BPUComponentConfig",
    "BranchPredictionConfig",
    "MicroBTBConfig",
]


@dataclass(frozen=True)
class BPUComponentConfig:
    """Configuration of a single sub-predictor of the branch prediction unit."""

    def meta_width(self, fetch_width: int) -> int:
        """Width of the metadata this component produces with every prediction and needs
        back at training time."""
        return 0

    def validate(self) -> None:
        """Raise ValueError if this component's parameters are incorrect."""


@dataclass(frozen=True)
class MicroBTBConfig(BPUComponentConfig):
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


@dataclass(frozen=True)
class BranchPredictionConfig:
    """Configuration of the branch prediction unit and all of its sub-predictors."""

    micro_btb: MicroBTBConfig = MicroBTBConfig()

    def components(self) -> tuple[BPUComponentConfig, ...]:
        return (self.micro_btb,)

    def bpd_meta_width(self, fetch_width: int) -> int:
        return sum(component.meta_width(fetch_width) for component in self.components())

    def validate(self):
        for component in self.components():
            component.validate()
