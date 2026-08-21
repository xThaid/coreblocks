from amaranth import *

from transactron.core import *
from transactron.utils.transactron_helpers import make_layout
from transactron.utils import logging
from transactron.lib import Pipe

from coreblocks.params import *
from coreblocks.arch import CfiType
from coreblocks.frontend import FrontendParams
from coreblocks.frontend.bpu.component import BPUComponentLayouts
from coreblocks.interface.layouts import CommonLayoutFields
from coreblocks.interface.layouts import BranchPredictionLayouts
from coreblocks.interface.layouts import FetchLayouts

log = logging.HardwareLogger("frontend.bpu")


class BranchPredictionUnit(Elaboratable):
    """Composable branch prediction unit.

    The composer knows nothing about the individual predictors: it broadcasts every
    request, collects each component's partial prediction and folds them into
    the combined prediction, with later components in the composition overriding earlier
    ones.

    The two outputs of a prediction are decoupled by design:

    - `write_fetch_target` supplies the next fetch target. It closes the fetch-address
      loop, so it carries nothing but the PC and must stay as fast as possible,
    - `write_prediction_details` delivers the full prediction for the fetch block
      together with the predictor metadata that the FTQ stores and hands back at
      training time.

    At training time the stored metadata is sliced back apart and each component
    receives exactly the bits it produced, in composition order.
    """

    request: Provided[Method]

    write_fetch_target: Required[Method]
    """Supplies the next fetch target."""

    write_prediction_details: Required[Method]
    """Delivers the full prediction for the fetch block together with the predictor metadata."""

    update: Provided[Method]
    flush: Provided[Method]

    def __init__(self, gen_params: GenParams) -> None:
        self.gen_params = gen_params
        self.layouts = gen_params.get(BranchPredictionLayouts)

        self.request = Method(i=self.layouts.request)
        self.write_fetch_target = Method(i=self.layouts.fetch_target)
        self.write_prediction_details = Method(i=self.layouts.prediction_details)
        self.update = Method(i=self.layouts.update)
        self.flush = Method()

    def elaborate(self, platform):
        m = TModule()

        fparams = self.gen_params.get(FrontendParams)
        fields = self.gen_params.get(CommonLayoutFields)
        fetch_layouts = self.gen_params.get(FetchLayouts)
        component_layouts = self.gen_params.get(BPUComponentLayouts)

        components = [config.get_module(self.gen_params) for config in self.gen_params.bpu_config.predictors]
        for i, component in enumerate(components):
            m.submodules[f"component_{i}"] = component
            if component.latency != 1:
                raise ValueError(
                    f"{type(component).__name__} answers in {component.latency} cycles, "
                    "but the BPU does not support it for now"
                )

        m.submodules.pipe = pipe = Pipe(
            layout=make_layout(fields.pc, fields.ftq_ptr, ("entry_idx", self.gen_params.fetch_width_log))
        )

        @def_method(m, self.request)
        def _(pc, ftq_ptr):
            for component in components:
                component.request(m, pc=pc)
            pipe.write(
                m,
                pc=fparams.pc_from_fb(fparams.fb_addr(pc) + 1, 0),
                ftq_ptr=ftq_ptr,
                entry_idx=fparams.fb_instr_idx(pc),
            )

        with Transaction(name="BPU_Stage1").body(m):
            partials = [component.predict(m) for component in components]
            stage = pipe.read(m)

            combined = Signal(component_layouts.combined_prediction)
            for partial in partials:
                with m.If(partial.hit & (partial.cfi_idx >= stage.entry_idx)):
                    m.d.av_comb += [
                        combined.hit.eq(1),
                        combined.cfi_target.eq(partial.cfi_target),
                        combined.cfi_idx.eq(partial.cfi_idx),
                        combined.cfi_type.eq(partial.cfi_type),
                    ]

            # On a hit, redirect fetch to the predicted target; otherwise fall through
            next_pc = Mux(combined.hit, combined.cfi_target, stage.pc)

            pred = Signal(fetch_layouts.bpu_prediction)
            m.d.av_comb += [
                pred.cfi_target.eq(combined.cfi_target),
                pred.cfi_target_valid.eq(combined.hit),
                pred.cfi_idx.eq(combined.cfi_idx),
                pred.cfi_type.eq(Mux(combined.hit, combined.cfi_type, CfiType.INVALID)),
                pred.branch_mask.eq(Mux(combined.hit & CfiType.is_branch(combined.cfi_type), 1 << combined.cfi_idx, 0)),
            ]
            self.write_fetch_target(m, pc=next_pc, ftq_ptr=stage.ftq_ptr)
            self.write_prediction_details(
                m, ftq_ptr=stage.ftq_ptr, prediction=pred, meta=Cat(partial.meta for partial in partials)
            )

        @def_method(m, self.update)
        def _(pc, cfi_target, cfi_idx, cfi_type, taken, mispredict, meta):
            offset = 0
            for component in components:
                component.update(
                    m,
                    pc=pc,
                    cfi_target=cfi_target,
                    cfi_idx=cfi_idx,
                    cfi_type=cfi_type,
                    taken=taken,
                    mispredict=mispredict,
                    meta=meta[offset : offset + component.meta_width],
                )
                offset += component.meta_width

        @def_method(m, self.flush, nonexclusive=True)
        def _():
            pipe.clear(m)

        return m
