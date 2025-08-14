# exp_timer.py
from datetime import datetime
import time
import logging
import functools
import threading
from types import FunctionType
from typing import Optional, Callable, Iterable

# Thread-local storage to keep track of depth
_local = threading.local()
timer_records = []

def _get_depth():
    return getattr(_local, "depth", 0)

def _inc_depth():
    _local.depth = _get_depth() + 1

def _dec_depth():
    _local.depth = max(0, _get_depth() - 1)

class TimerConfig:
    def __init__(
        self,
        log_file: Optional[str] = None,
        enabled: bool = True,
        threshold_ms: float = 0.0,  # Default: log all
        fmt: str = "[%(asctime)s] %(message)s",
        datefmt: str = "%H:%M:%S",
    ):
        self.enabled = enabled
        self.threshold_ms = threshold_ms

        self.logger = logging.getLogger("exp_timer")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            self.logger.addHandler(sh)
        
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            self.logger.addHandler(fh)

# default config for the single case
CONFIG = TimerConfig()

def _log(msg: str):
    if CONFIG.enabled:
        CONFIG.logger.info(msg)

def set_timer_context(**kwargs):
    for k, v in kwargs.items():
        setattr(_local, k, v)

def time_block(label: str, *, extra: str = ""):
    """Timer for a code block
    Usage:
        with time_block("stage1"):
            ...
    """
    class _Ctx:
        def __enter__(self):
            self.t0 = time.perf_counter()
            _inc_depth()
            return self

        def __exit__(self, exc_type, exc, tb):
            t1 = time.perf_counter()
            _dec_depth()
            elapsed_ms = (t1 - self.t0) * 1000.0
            if elapsed_ms >= CONFIG.threshold_ms:
                indent = "  " * _get_depth()
                tag = f" [{extra}]" if extra else ""
                _log(f"{indent} {label}{tag} took {elapsed_ms:.3f} ms")

            return False
    return _Ctx()

def timer(name: Optional[str] = None, *, extra: None, threshold_ms: Optional[float] = None):
    """Function/method decorator for timing
    - name: Display name (default is qualname)
    - extra: Additional tag (e.g. dataset name/experiment group)
    - threshold_ms: Override global threshold, only log if exceeds this duration
    """
    def deco(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not CONFIG.enabled:
                return func(*args, **kwargs)

            label = name or getattr(func, "__qualname__", func.__name__)
            t0 = time.perf_counter()
            _inc_depth()
            try:
                return func(*args, **kwargs)
            finally:
                t1 = time.perf_counter()
                _dec_depth()
                elapsed_ms = (t1 - t0) * 1000.0
                th = CONFIG.threshold_ms if threshold_ms is None else threshold_ms
                if elapsed_ms >= th:
                    indent = "  " * _get_depth()
                    actual_extra = extra(_local) if callable(extra) else extra
                    tag = f" [{actual_extra}]" if actual_extra else ""
                    _log(f"{indent} {label}{tag} took {elapsed_ms:.3f} ms")
                    log_timer_result(
                        label=label,
                        duration_ms=elapsed_ms,
                    )
        # Mark the function as timed
        wrapper._is_timed = True
        return wrapper
    return deco

def log_timer_result(label, duration_ms):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": getattr(_local, 'exp_id', None),
        "trial": getattr(_local, 'trial', None),
        "process": label,
        "k": getattr(_local, 'k', None),
        "duration_ms": round(duration_ms, 3),
    }
    timer_records.append(record)


def _wrap_any_method(attr_value, wrapper_factory):
    """Support regular functions, staticmethods, and classmethods while preserving their types"""
    if isinstance(attr_value, staticmethod):
        f = attr_value.__func__
        return staticmethod(wrapper_factory(f))
    elif isinstance(attr_value, classmethod):
        f = attr_value.__func__
        return classmethod(wrapper_factory(f))
    elif isinstance(attr_value, FunctionType):
        return wrapper_factory(attr_value)
    else:
        return attr_value  # Not a callable function: keep as it is
    
def _has_timer(attr) -> bool:
    # Regular function or already wrapped function
    if hasattr(attr, "_is_timed"):
        return True
    # staticmethod / classmethod: Check the underlying function
    if isinstance(attr, (staticmethod, classmethod)):
        return hasattr(attr.__func__, "_is_timed")
    return False

def time_all_methods(
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Iterable[str] = ("__init__", "__repr__", "__str__", "__class__", "__new__", "__getattr__", "__getattribute__"),
    extra: str = "",
    threshold_ms: Optional[float] = None,
):
    """Class decorator for automatically timing methods
    - include: Only time methods with these names (empty means auto-discover)
    - exclude: Methods to exclude: default excludes common dunder methods
    The following arguments are passed to the timer decorator: 
    - extra: Additional tag (e.g. dataset name/experiment group)
    - threshold_ms: Override global threshold, only log if exceeds this duration
    """
    exclude = set(exclude or ()) 

    def class_decorator(cls):
        names = include if include is not None else list(cls.__dict__.keys())
        for name in names:
            if name in exclude:
                continue
            attr_value = cls.__dict__.get(name)
            if (callable(attr_value) or isinstance(attr_value, (staticmethod, classmethod))) and not _has_timer(attr_value): 
                def factory(f):
                    return timer(None, extra=extra, threshold_ms=threshold_ms)(f)
                wrapped = _wrap_any_method(attr_value, factory)
                setattr(cls, name, wrapped)
        return cls
    return class_decorator
