from amaranth import *

from transactron import *
from transactron.utils.transactron_helpers import make_layout

from coreblocks.params import GenParams
from coreblocks.interface.layouts import CommonLayoutFields

__all__ = ["BPUComponentLayouts", "BPUComponent", "TargetPredictor"]


class BPUComponentLayouts:
    """Uniform layouts of the branch predictor component interface."""

    def __init__(self, gen_params: GenParams):
        fields = gen_params.get(CommonLayoutFields)

        self.request = make_layout(fields.pc)

        self._target_prediction_fields = [("hit", 1), fields.cfi_target, fields.cfi_idx, fields.cfi_type]
        self._update_fields = [
            fields.pc,
            fields.cfi_target,
            fields.cfi_idx,
            fields.cfi_type,
            ("taken", 1),
            ("mispredict", 1),
        ]

    def target_prediction(self, meta_width: int):
        return make_layout(*self._target_prediction_fields, ("meta", meta_width))

    def update(self, meta_width: int):
        return make_layout(*self._update_fields, ("meta", meta_width))


class BPUComponent(Elaboratable):
    """Base class of the branch predictor components.

    Attributes
    ----------
    latency : int
        Number of cycles between ``request`` and the component's ``predict`` result being
        ready. The BPU calls ``predict`` exactly at that stage.
    meta_width : int
        Width of this component's prediction metadata.
    """

    latency: int = 1

    request: Provided[Method]
    """Start a lookup for a fetch block PC."""
    predict: Provided[Method]
    """Return the component's prediction, ``latency`` cycles after ``request``."""
    update: Provided[Method]
    """Train the component with a resolved CFI and its own slice of the stored prediction
    metadata."""

    def __init__(self, gen_params: GenParams, meta_width: int):
        self.gen_params = gen_params
        self.meta_width = meta_width
        self.component_layouts = gen_params.get(BPUComponentLayouts)

        self.request = Method(i=self.component_layouts.request)
        self.update = Method(i=self.component_layouts.update(meta_width))

    def elaborate(self, platform) -> TModule:
        raise NotImplementedError()


class TargetPredictor(BPUComponent):
    """A component that identifies the fetch block's CFI: its position, type and target."""

    def __init__(self, gen_params: GenParams, meta_width: int):
        super().__init__(gen_params, meta_width)
        self.predict = Method(o=self.component_layouts.target_prediction(meta_width))
