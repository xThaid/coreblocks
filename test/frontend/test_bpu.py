import pytest
from collections import deque
from dataclasses import dataclass

from amaranth import *

from transactron.testing import (
    TestCaseWithSimulator,
    SimpleTestCircuit,
    TestbenchContext,
    def_method_mock,
)
from transactron.testing.method_mock import MethodMock

from transactron import TModule, def_method

from coreblocks.arch import CfiType
from coreblocks.frontend.bpu.bpu import BranchPredictionUnit
from coreblocks.frontend.bpu.component import TargetPredictor
from coreblocks.params import BPUPredictorConfig, BranchPredictionConfig, GenParams
from coreblocks.params import configurations


class StubPredictor(TargetPredictor):
    def __init__(self, gen_params: GenParams, config: "StubPredictorConfig"):
        super().__init__(gen_params, meta_width=config.meta_width(gen_params.fetch_width))
        self.config = config

    def elaborate(self, platform):
        m = TModule()

        echo = Signal(self.gen_params.isa.xlen)

        @def_method(m, self.request)
        def _(pc):
            pass

        @def_method(m, self.predict)
        def _():
            return {
                "hit": 1,
                "cfi_target": echo,
                "cfi_idx": 0,
                "cfi_type": CfiType.BRANCH,
                "meta": C(self.config.meta, self.meta_width),
            }

        @def_method(m, self.update)
        def _(pc, cfi_target, cfi_idx, cfi_type, taken, mispredict, meta):
            m.d.sync += echo.eq(meta)

        return m


@dataclass(frozen=True)
class StubPredictorConfig(BPUPredictorConfig):
    meta: int = 0
    meta_bits: int = 8

    def meta_width(self, fetch_width: int) -> int:
        return self.meta_bits

    def get_module(self, gen_params: GenParams) -> StubPredictor:
        return StubPredictor(gen_params, self)


class TestBranchPredictionUnit(TestCaseWithSimulator):
    @pytest.fixture(autouse=True)
    def setup(self, fixture_initialize_testing_env):
        # A multi-instruction fetch block, so in-block CFI indices are meaningful
        self.gen_params = GenParams(configurations.test.replace(fetch_block_bytes_log=4))
        self.fbl = self.gen_params.fetch_block_bytes_log
        self.bpu = SimpleTestCircuit(BranchPredictionUnit(self.gen_params))
        self.fetch_targets: deque = deque()
        self.predictions: deque = deque()

    def fall_through(self, pc: int) -> int:
        return ((pc >> self.fbl) + 1) << self.fbl

    @def_method_mock(lambda self: self.bpu.write_fetch_target)
    def write_fetch_target_mock(self, pc, ftq_ptr):
        @MethodMock.effect
        def eff():
            self.fetch_targets.append(pc)

    @def_method_mock(lambda self: self.bpu.write_prediction_details)
    def write_prediction_details_mock(self, ftq_ptr, prediction, meta):
        @MethodMock.effect
        def eff():
            self.predictions.append(prediction)

    async def predict(self, sim: TestbenchContext, pc: int) -> int:
        self.fetch_targets.clear()
        self.predictions.clear()
        await self.bpu.request.call(sim, pc=pc, ftq_ptr={"ptr": 0, "parity": 0})
        while not self.fetch_targets:
            await sim.tick()
        assert len(self.predictions) == len(self.fetch_targets)
        return self.fetch_targets[-1]

    def test_unknown_block_falls_through(self):
        pc = 0x100

        async def proc(sim: TestbenchContext):
            assert await self.predict(sim, pc) == self.fall_through(pc)

        with self.run_simulation(self.bpu) as sim:
            sim.add_testbench(proc)

    def test_learned_taken_branch_redirects_to_target(self):
        pc = 0x100
        target = 0x2ABC

        async def proc(sim: TestbenchContext):
            assert await self.predict(sim, pc) == self.fall_through(pc)

            await self.bpu.update.call(sim, pc=pc, cfi_target=target, cfi_idx=0, cfi_type=CfiType.BRANCH, taken=1)

            assert await self.predict(sim, pc) == target

        with self.run_simulation(self.bpu) as sim:
            sim.add_testbench(proc)

    def test_not_taken_update_does_not_redirect(self):
        pc = 0x100

        async def proc(sim: TestbenchContext):
            await self.bpu.update.call(sim, pc=pc, cfi_target=0xDEAD, cfi_idx=0, cfi_type=CfiType.BRANCH, taken=0)
            assert await self.predict(sim, pc) == self.fall_through(pc)

        with self.run_simulation(self.bpu) as sim:
            sim.add_testbench(proc)


class TestBranchPredictionUnitComposition(TestCaseWithSimulator):
    @pytest.fixture(autouse=True)
    def setup(self, fixture_initialize_testing_env):
        self.base_config = configurations.test.replace(fetch_block_bytes_log=4)
        self.fetch_targets: deque = deque()
        self.metas: deque = deque()

    def build(self, *predictors: BPUPredictorConfig):
        gen_params = GenParams(self.base_config.replace(bpu_config=BranchPredictionConfig(predictors=predictors)))
        self.fbl = gen_params.fetch_block_bytes_log
        self.bpu = SimpleTestCircuit(BranchPredictionUnit(gen_params))
        return self.bpu

    def fall_through(self, pc: int) -> int:
        return ((pc >> self.fbl) + 1) << self.fbl

    @def_method_mock(lambda self: self.bpu.write_fetch_target)
    def write_fetch_target_mock(self, pc, ftq_ptr):
        @MethodMock.effect
        def eff():
            self.fetch_targets.append(pc)

    @def_method_mock(lambda self: self.bpu.write_prediction_details)
    def write_prediction_details_mock(self, ftq_ptr, prediction, meta):
        @MethodMock.effect
        def eff():
            self.metas.append(meta)

    async def predict(self, sim: TestbenchContext, pc: int) -> int:
        self.fetch_targets.clear()
        await self.bpu.request.call(sim, pc=pc, ftq_ptr={"ptr": 0, "parity": 0})
        while not self.fetch_targets:
            await sim.tick()
        return self.fetch_targets[-1]

    def test_empty_composition_always_falls_through(self):
        bpu = self.build()

        async def proc(sim: TestbenchContext):
            assert await self.predict(sim, 0x100) == self.fall_through(0x100)

        with self.run_simulation(bpu) as sim:
            sim.add_testbench(proc)

    def test_later_component_overrides_earlier_one(self):
        bpu = self.build(
            StubPredictorConfig(meta=0x5, meta_bits=4),
            StubPredictorConfig(meta=0xAB, meta_bits=8),
        )

        async def proc(sim: TestbenchContext):
            await self.bpu.update.call(
                sim,
                pc=0x100,
                cfi_target=0,
                cfi_idx=0,
                cfi_type=CfiType.BRANCH,
                taken=1,
                mispredict=0,
                meta=(0xAB << 4) | 0x5,
            )
            assert await self.predict(sim, 0x100) == 0xAB

        with self.run_simulation(bpu) as sim:
            sim.add_testbench(proc)

    def test_metadata_is_concatenated_in_composition_order(self):
        bpu = self.build(
            StubPredictorConfig(meta=0x5, meta_bits=4),
            StubPredictorConfig(meta=0xAB, meta_bits=8),
        )

        async def proc(sim: TestbenchContext):
            self.metas.clear()
            await self.predict(sim, 0x100)
            assert self.metas[-1] == (0xAB << 4) | 0x5

        with self.run_simulation(bpu) as sim:
            sim.add_testbench(proc)

    def test_single_component_composition_uses_its_prediction(self):
        bpu = self.build(StubPredictorConfig(meta=0x7C, meta_bits=8))

        async def proc(sim: TestbenchContext):
            await self.bpu.update.call(
                sim,
                pc=0x100,
                cfi_target=0,
                cfi_idx=0,
                cfi_type=CfiType.BRANCH,
                taken=1,
                mispredict=0,
                meta=0x7C,
            )
            assert await self.predict(sim, 0x100) == 0x7C

        with self.run_simulation(bpu) as sim:
            sim.add_testbench(proc)
