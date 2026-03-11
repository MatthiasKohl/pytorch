"""Shared types and constants for parse_sass.py and plot_sass_summary.py."""

from dataclasses import dataclass

DTYPE_SIZES = {
    "bool": 8,
    "c10::BFloat16": 16,
    "c10::Float4_e2m1fn_x2": 4,
    "c10::Float8_e4m3fn": 8,
    "c10::Float8_e4m3fnuz": 8,
    "c10::Float8_e5m2": 8,
    "c10::Float8_e5m2fnuz": 8,
    "c10::Float8_e8m0fnu": 8,
    "c10::Half": 16,
    "c10::bits16": 16,
    "c10::bits1x8": 8,
    "c10::bits2x4": 8,
    "c10::bits4x2": 8,
    "c10::bits8": 8,
    "c10::complex<c10::Half>": 32,
    "c10::complex<double>": 128,
    "c10::complex<float>": 64,
    "c10::qint32": 32,
    "c10::qint8": 8,
    "c10::quint8": 8,
    "double": 64,
    "float": 32,
    "int": 32,
    "long": 64,
    "short": 16,
    "signed char": 8,
    "unsigned char": 8,
    "unsigned int": 32,
    "unsigned long": 64,
    "unsigned short": 16,
}


@dataclass
class KernelStats:
    mangled: str
    demangled: str
    tag: int
    functor: str
    scalar_type: str
    source_file: str
    num_instructions: int = 0
    max_gpr: int = 0
    min_gpr: int = 0
    max_pred: int = 0
    min_pred: int = 0
    max_ugpr: int = 0
    min_ugpr: int = 0

    @property
    def gpr_range(self) -> int:
        return self.max_gpr - self.min_gpr

    @property
    def pred_range(self) -> int:
        return self.max_pred - self.min_pred

    @property
    def ugpr_range(self) -> int:
        return self.max_ugpr - self.min_ugpr

    @property
    def line(self) -> int:
        return self.tag
