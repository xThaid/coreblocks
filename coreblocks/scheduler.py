from amaranth import *
from coreblocks.transactions import Method, Transaction
from coreblocks.transactions.lib import FIFO
from coreblocks.layouts import SchedulerLayouts, ROBLayouts
from coreblocks.genparams import GenParams
from coreblocks.utils import assign

__all__ = ["Scheduler"]


class RegAllocation(Elaboratable):
    def __init__(self, *, get_instr: Method, push_instr: Method, get_free_reg: Method, gen_params: GenParams):
        self.gen_params = gen_params
        layouts = gen_params.get(SchedulerLayouts)
        self.input_layout = layouts.reg_alloc_in
        self.output_layout = layouts.reg_alloc_out

        self.get_instr = get_instr
        self.push_instr = push_instr
        self.get_free_reg = get_free_reg

    def elaborate(self, platform):
        m = Module()

        free_reg = Signal(self.gen_params.phys_regs_bits)
        data_out = Record(self.output_layout)

        with Transaction().body(m):
            instr = self.get_instr(m)
            with m.If(instr.rl_dst != 0):
                reg_id = self.get_free_reg(m)
                m.d.comb += free_reg.eq(reg_id)

            m.d.comb += assign(data_out, instr)
            m.d.comb += data_out.rp_dst.eq(free_reg)
            self.push_instr(m, data_out)

        return m


class Renaming(Elaboratable):
    def __init__(self, *, get_instr: Method, push_instr: Method, rename: Method, gen_params: GenParams):
        self.gen_params = gen_params
        layouts = gen_params.get(SchedulerLayouts)
        self.input_layout = layouts.renaming_in
        self.output_layout = layouts.renaming_out

        self.get_instr = get_instr
        self.push_instr = push_instr
        self.rename = rename

    def elaborate(self, platform):
        m = Module()

        data_out = Record(self.output_layout)

        with Transaction().body(m):
            instr = self.get_instr(m)
            renamed_regs = self.rename(m, instr)

            m.d.comb += assign(data_out, instr, fields={"rl_dst", "rp_dst", "opcode"})
            m.d.comb += assign(data_out, renamed_regs)
            self.push_instr(m, data_out)

        return m


class ROBAllocation(Elaboratable):
    def __init__(self, *, get_instr: Method, push_instr: Method, rob_put: Method, gen_params: GenParams):
        self.gen_params = gen_params
        layouts = gen_params.get(SchedulerLayouts)
        self.input_layout = layouts.rob_allocate_in
        self.output_layout = layouts.rob_allocate_out
        self.rob_req_layout = gen_params.get(ROBLayouts).data_layout

        self.get_instr = get_instr
        self.push_instr = push_instr
        self.rob_put = rob_put

    def elaborate(self, platform):
        m = Module()

        data_out = Record(self.output_layout)
        rob_req = Record(self.rob_req_layout)

        with Transaction().body(m):
            instr = self.get_instr(m)

            m.d.comb += assign(rob_req, instr, fields={"rl_dst", "rp_dst"})
            rob_id = self.rob_put(m, rob_req)

            m.d.comb += assign(data_out, instr, fields={"rp_s1", "rp_s2", "rp_dst", "opcode"})
            m.d.comb += data_out.rob_id.eq(rob_id.rob_id)

            self.push_instr(m, data_out)

        return m


class RSSelection(Elaboratable):
    def __init__(self, *, get_instr: Method, push_instr: Method, rs_alloc: Method, gen_params: GenParams):
        self.gen_params = gen_params
        layouts = gen_params.get(SchedulerLayouts)
        self.input_layout = layouts.rs_select_in
        self.output_layout = layouts.rs_select_out

        self.get_instr = get_instr
        self.push_instr = push_instr
        self.rs_alloc = rs_alloc

    def elaborate(self, platform):
        m = Module()

        with Transaction().body(m):
            instr = self.get_instr(m)
            allocated_field = self.rs_alloc(m)
            self.push_instr(
                m,
                {
                    "rp_s1": instr.rp_s1,
                    "rp_s2": instr.rp_s2,
                    "rp_dst": instr.rp_dst,
                    "rob_id": instr.rob_id,
                    "opcode": instr.opcode,
                    "rs_entry_id": allocated_field.entry_id,
                },
            )

        return m


class RSInsertion(Elaboratable):
    def __init__(
        self, *, get_instr: Method, rs_insert: Method, rf_read1: Method, rf_read2: Method, gen_params: GenParams
    ):
        self.gen_params = gen_params

        self.get_instr = get_instr
        self.rs_insert = rs_insert
        self.rf_read1 = rf_read1
        self.rf_read2 = rf_read2

    def elaborate(self, platform):
        m = Module()

        with Transaction().body(m):
            instr = self.get_instr(m)
            source1 = self.rf_read1(m, {"reg_id": instr.rp_s1})
            source2 = self.rf_read2(m, {"reg_id": instr.rp_s2})

            self.rs_insert(
                m,
                {
                    # when operand value is valid the convention is to set operand source to 0
                    "rp_s1": Mux(source1.valid, 0, instr.rp_s1),
                    "rp_s2": Mux(source2.valid, 0, instr.rp_s2),
                    "rp_dst": instr.rp_dst,
                    "rob_id": instr.rob_id,
                    "opcode": instr.opcode,
                    "rs_entry_id": instr.rs_entry_id,
                    "s1_val": Mux(source1.valid, source1.reg_val, 0),
                    "s2_val": Mux(source2.valid, source2.reg_val, 0),
                },
            )

        return m


class Scheduler(Elaboratable):
    def __init__(
        self,
        *,
        get_instr: Method,
        get_free_reg: Method,
        rat_rename: Method,
        rob_put: Method,
        rf_read1: Method,
        rf_read2: Method,
        rs_alloc: Method,
        rs_insert: Method,
        gen_params: GenParams
    ):
        self.gen_params = gen_params
        self.layouts = self.gen_params.get(SchedulerLayouts)
        self.get_instr = get_instr
        self.get_free_reg = get_free_reg
        self.rat_rename = rat_rename
        self.rob_put = rob_put
        self.rf_read1 = rf_read1
        self.rf_read2 = rf_read2
        self.rs_alloc = rs_alloc
        self.rs_insert = rs_insert

    def elaborate(self, platform):
        m = Module()

        m.submodules.alloc_rename_buf = alloc_rename_buf = FIFO(self.layouts.reg_alloc_out, 2)
        m.submodules.reg_alloc = RegAllocation(
            get_instr=self.get_instr,
            push_instr=alloc_rename_buf.write,
            get_free_reg=self.get_free_reg,
            gen_params=self.gen_params,
        )

        m.submodules.rename_out_buf = rename_out_buf = FIFO(self.layouts.renaming_out, 2)
        m.submodules.renaming = Renaming(
            get_instr=alloc_rename_buf.read,
            push_instr=rename_out_buf.write,
            rename=self.rat_rename,
            gen_params=self.gen_params,
        )

        m.submodules.reg_alloc_out_buf = reg_alloc_out_buf = FIFO(self.layouts.rob_allocate_out, 2)
        m.submodules.rob_alloc = ROBAllocation(
            get_instr=rename_out_buf.read,
            push_instr=reg_alloc_out_buf.write,
            rob_put=self.rob_put,
            gen_params=self.gen_params,
        )

        m.submodules.rs_select_out_buf = rs_select_out_buf = FIFO(self.layouts.rs_select_out, 2)
        m.submodules.rs_select = RSSelection(
            get_instr=reg_alloc_out_buf.read,
            push_instr=rs_select_out_buf.write,
            rs_alloc=self.rs_alloc,
            gen_params=self.gen_params,
        )

        m.submodules.rs_insert = RSInsertion(
            get_instr=rs_select_out_buf.read,
            rs_insert=self.rs_insert,
            rf_read1=self.rf_read1,
            rf_read2=self.rf_read2,
            gen_params=self.gen_params,
        )

        return m