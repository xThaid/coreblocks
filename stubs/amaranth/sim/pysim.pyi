"""
This type stub file was generated by pyright.
"""

from contextlib import contextmanager
from ..hdl import *
from ._base import *

__all__ = ["PySimEngine"]
class _NameExtractor:
    def __init__(self) -> None:
        ...
    
    def __call__(self, fragment, *, hierarchy=...): # -> SignalDict:
        ...
    


class _VCDWriter:
    @staticmethod
    def decode_to_vcd(signal, value):
        ...
    
    def __init__(self, fragment, *, vcd_file, gtkw_file=..., traces=...) -> None:
        ...
    
    def update(self, timestamp, signal, value): # -> None:
        ...
    
    def close(self, timestamp):
        ...
    
    vcd_file: Any
    gtkw_file: Any
    gtkw_save: Any
    gtkw_names: Any
    vcd_writer: Any

class _Timeline:
    def __init__(self) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def at(self, run_at, process): # -> None:
        ...
    
    def delay(self, delay_by, process): # -> None:
        ...
    
    def advance(self): # -> bool:
        ...
    


class _PySignalState(BaseSignalState):
    __slots__ = ...
    def __init__(self, signal, pending) -> None:
        ...
    
    def set(self, value): # -> None:
        ...
    
    def commit(self): # -> bool:
        ...
    


class _PySimulation(BaseSimulation):
    def __init__(self) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def get_signal(self, signal): # -> int:
        ...
    
    def add_trigger(self, process, signal, *, trigger=...): # -> None:
        ...
    
    def remove_trigger(self, process, signal): # -> None:
        ...
    
    def wait_interval(self, process, interval): # -> None:
        ...
    
    def commit(self, changed=...): # -> bool:
        ...
    


class PySimEngine(BaseEngine):
    def __init__(self, fragment) -> None:
        ...
    
    def add_coroutine_process(self, process, *, default_cmd): # -> None:
        ...
    
    def add_clock_process(self, clock, *, phase, period): # -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def advance(self): # -> bool:
        ...
    
    @property
    def now(self): # -> int | None:
        ...
    
    @contextmanager
    def write_vcd(self, *, vcd_file, gtkw_file, traces): # -> Generator[None, None, None]:
        ...
    


