"""Bit-exact reference for the bf16_fp32_ops RTL: BF16 x BF16 -> binary32 multiply and binary32 add.

Two independent models, standard library only:

* ``mul_bf16`` / ``add32`` (the golden): integer dyadic arithmetic, one round-to-nearest-even (RNE) per
  operation, gradual underflow, no host floating point. Derived from ChipLab's internal
  floating-point reference; see PROVENANCE.md.
* ``oracle_mul_bf16`` / ``oracle_add32``: host IEEE binary64 arithmetic followed by one conversion to
  binary32. A BF16 x BF16 product is exact in binary64, and binary64 is wide enough (53 >= 2*24 + 2) that
  rounding an add twice (to binary64, then binary32) equals rounding it once. The test bench requires both
  models to agree on every vector before the RTL is compared against them.

Numeric behavior (what the RTL implements):
  * RNE, gradual underflow (subnormal inputs and outputs are honored, never flushed).
  * Overflow gives a signed infinity.
  * Any NaN input, inf x 0 and inf - inf give the canonical quiet NaN 0x7fc00000 (payload and sign dropped).
  * Exact cancellation in an add gives +0; only (-0) + (-0) gives -0. A zero product has sign a ^ b.
  * No exception flags are produced.
"""
from __future__ import annotations

import math
import struct

SIGN = 0x80000000
INF = 0x7F800000
QNAN = 0x7FC00000


def classification(value: int) -> str:
    exponent, fraction = (value >> 23) & 255, value & 0x7FFFFF
    if exponent == 255:
        return "nan" if fraction else "infinity"
    if exponent == 0:
        return "subnormal" if fraction else "zero"
    return "normal"


def _finite(value: int) -> tuple[int, int]:
    exponent, fraction = (value >> 23) & 255, value & 0x7FFFFF
    mantissa = fraction if exponent == 0 else (1 << 23) | fraction
    return (-mantissa if value & SIGN else mantissa,
            -149 if exponent == 0 else exponent - 150)


def _round_even(value: int, shift: int) -> int:
    if shift <= 0:
        return value << -shift
    whole, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    return whole + int(remainder > half or (remainder == half and whole & 1))


def _pack(mantissa: int, exponent: int, zero_sign: int = 0) -> int:
    """Round exact mantissa * 2**exponent once to binary32, RNE."""
    if not mantissa:
        return zero_sign & SIGN
    sign = SIGN if mantissa < 0 else 0
    magnitude = abs(mantissa)
    top_exp = magnitude.bit_length() - 1 + exponent
    quantum = max(top_exp - 23, -149)
    rounded = _round_even(magnitude, quantum - exponent)
    if not rounded:
        return sign
    if rounded < (1 << 23):
        return sign | rounded
    if rounded == (1 << 24):
        rounded >>= 1
        quantum += 1
    encoded_exp = quantum + 23 + 127
    if encoded_exp >= 255:
        return sign | INF
    return sign | (encoded_exp << 23) | (rounded - (1 << 23))


def add32(a: int, b: int) -> int:
    """binary32 + binary32 -> binary32 (golden)."""
    ca, cb = classification(a), classification(b)
    if "nan" in (ca, cb):
        return QNAN
    if ca == "infinity" or cb == "infinity":
        if ca == cb and (a ^ b) & SIGN:
            return QNAN
        return a if ca == "infinity" else b
    ma, ea = _finite(a)
    mb, eb = _finite(b)
    exponent = min(ea, eb)
    exact = (ma << (ea - exponent)) + (mb << (eb - exponent))
    zero_sign = SIGN if a == SIGN and b == SIGN else 0
    return _pack(exact, exponent, zero_sign)


def mul32(a: int, b: int) -> int:
    """binary32 x binary32 -> binary32 (golden)."""
    ca, cb = classification(a), classification(b)
    sign = (a ^ b) & SIGN
    if "nan" in (ca, cb):
        return QNAN
    if "infinity" in (ca, cb):
        if "zero" in (ca, cb):
            return QNAN
        return sign | INF
    ma, ea = _finite(a)
    mb, eb = _finite(b)
    return _pack(ma * mb, ea + eb, sign)


def mul_bf16(a: int, b: int) -> int:
    """BF16 x BF16 -> binary32 (golden). A BF16 word is the top half of a binary32 word."""
    return mul32((a & 0xFFFF) << 16, (b & 0xFFFF) << 16)


# ---- independent host-float oracle ----

def _f32_to_float(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def _float_to_f32(x: float) -> int:
    if math.isnan(x):
        return QNAN
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]   # one RNE conversion
    except OverflowError:                                     # rounds beyond the binary32 range
        return (SIGN if x < 0 else 0) | INF


def oracle_mul_bf16(a: int, b: int) -> int:
    fa, fb = _f32_to_float((a & 0xFFFF) << 16), _f32_to_float((b & 0xFFFF) << 16)
    if math.isnan(fa) or math.isnan(fb):
        return QNAN
    return _float_to_f32(fa * fb)          # exact in binary64 (8-bit x 8-bit significands)


def oracle_add32(a: int, b: int) -> int:
    fa, fb = _f32_to_float(a), _f32_to_float(b)
    if math.isnan(fa) or math.isnan(fb):
        return QNAN
    return _float_to_f32(fa + fb)
