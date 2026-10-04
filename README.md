# bf16_fp32_ops: BF16 x BF16 -> FP32 multiplier and FP32 adder (synthesizable SystemVerilog)

Three small, dependency-free SystemVerilog modules that implement the arithmetic of a BF16-multiply,
FP32-accumulate (BF16-WA) MAC, checked bit-exact against a Python reference:

- `a50_bf16_mul`: BF16 x BF16 -> binary32, combinational.
- `a50_add32_align` + `a50_round_pack`: binary32 + binary32 -> binary32, split into two combinational stages
  so a register can be placed between them (align/add, then normalize/round).

There is no HardFloat or vendor IP; it is plain RTL that Icarus Verilog, Verilator and Yosys read.
These operators come from ChipLab, a hobby AI-accelerator project run as a lab of AI coding agents (Claude and Codex)
directed by its owner; the RTL, models and tests here were written by those agents. ChipLab uses BF16 multiply with FP32
accumulate. Provenance is in [PROVENANCE.md](PROVENANCE.md).

## Interface

### `a50_bf16_mul` (combinational)

| port | dir | width | meaning |
|---|---|---|---|
| `a` | in | 16 | BF16 operand (sign, 8-bit exponent, 7-bit fraction) |
| `b` | in | 16 | BF16 operand |
| `p` | out | 32 | binary32 product, rounded once |

### `a50_add32_align` (combinational, adder stage 1)

| port | dir | width | meaning |
|---|---|---|---|
| `a`, `b` | in | 32 | binary32 operands |
| `special` | out | 1 | 1 when the result is `special_word` (NaN or infinity) rather than a rounded sum |
| `special_word` | out | 32 | `0x7fc00000` or the infinite operand |
| `sign` | out | 1 | sign of the sum (applies the zero-sign rule below) |
| `sum` | out | 28 | aligned magnitude sum, with a jammed sticky bit in bit 0 |
| `lsb_exp` | out | 12 (signed) | binary exponent of `sum` bit 0 |

### `a50_round_pack` (combinational, adder stage 2; also used inside the multiplier)

| port | dir | width | meaning |
|---|---|---|---|
| `sign` | in | 1 | result sign |
| `sig` | in | 28 | magnitude; value = `sig * 2^lsb_exp` |
| `lsb_exp` | in | 12 (signed) | exponent of `sig` bit 0 |
| `word` | out | 32 | binary32, rounded once |

Adder result: `word = special ? special_word : round_pack(sign, sum, lsb_exp)`.
Connect the stages as `tb/tb_bf16_fp32_ops.sv` does, optionally with a register between them.
`a50_round_pack` assumes that a sticky bit jammed into `sig[0]` sits at least two bits below the rounding
position whenever it is set; `a50_add32_align` guarantees this (module header comments).

## Numeric behavior

Both operations match the reference in `model/bf16_fp32_ref.py` bit for bit:

| case | behavior |
|---|---|
| rounding | one round-to-nearest, ties-to-even per operation |
| subnormals | honored on input and output (gradual underflow, no flush-to-zero) |
| BF16 x BF16 | the 8x8-bit significand product is exact, so rounding happens only when the result underflows the binary32 normal range |
| overflow | signed infinity |
| NaN | any NaN input, inf x 0, and inf + (-inf) give the canonical quiet NaN `0x7fc00000`; NaN payload and sign are not propagated |
| zero sign | a zero product has sign `a ^ b`; an exactly cancelling sum is +0; only (-0) + (-0) gives -0 |
| flags | no IEEE exception flags (inexact, underflow, overflow, invalid) are produced |

## How to run

Requires Python 3.9+ (standard library only), GNU make and Icarus Verilog (`apt install iverilog`).
Verilator is optional.

```sh
make test        # full vector set on Icarus Verilog
make quick       # smaller vector set
make mutants     # each seeded bug must FAIL the bench (mutations are made in build/, not in rtl/)
make lint        # verilator --lint-only -Wall
make verilator   # full vector set on Verilator
```

`run_tests.py` (called by make) does three things:

1. It generates deterministic vectors (seed 2026): all pairs of a 104-value BF16 special-value grid,
   every one of the 65,536 BF16 values times six fixed operands, uniform random pairs, products aimed at the
   underflow boundary, all pairs of a 140-value binary32 grid, and random adds aimed at alignment and sticky
   bits, exact ties, cancellation, overflow and subnormals.
2. It computes every expected word with the integer golden model and requires an independent host-float oracle
   (binary64 arithmetic, then one conversion to binary32) to agree on every vector.
3. It simulates the RTL and accepts only one complete, self-consistent result line: both vector populations
   equal to the generated counts, both mismatch counts zero, and PASS. A seeded bug counts as killed only when that
   line reports the full populations, FAIL and at least one mismatch. A missing result, an `ERROR` line, an empty or
   partial run, or contradictory lines are neither a pass nor a kill. Fifteen fixed transcripts (parser controls)
   check this gate on every run before any simulation.

Results recorded when the package was created (Icarus Verilog 12.0, Verilator 5.020, Python 3.11; the counts
are reproducible with the commands above):

| command | result |
|---|---|
| parser controls (every run) | 15 cases, all as expected |
| `make test` | `RESULT mul=604032 mul_bad=0 add=279600 add_bad=0 PASS`, golden vs oracle mismatches 0 |
| `make verilator` | `RESULT mul=604032 mul_bad=0 add=279600 add_bad=0 PASS` |
| `make mutants` | 6/6 seeded bugs killed (ties-away rounding, overflow threshold, no sticky bit, zero-sign rule, inf x 0, flushed subnormal inputs) |
| `make lint` | Verilator `-Wall` clean |

In ChipLab's own CI, `make test` and `make mutants` run on every change. Run `make lint` and
`make verilator` locally (they need Verilator).

## What is not claimed

- **Area, timing, power and energy: none.** There is no committed mapped or routed result for these three modules
  as a standalone block, so this package states no figure.
- The tests are not exhaustive over all 2^32 multiplier or 2^64 adder input pairs. Coverage is the directed grids,
  the one-operand sweeps and the random sets described above.
- Correctness is relative to the reference in `model/`, which is RNE IEEE 754 binary32 arithmetic with the NaN,
  zero-sign and no-flag choices in the table above. It is not a formal proof and not a check against another
  vendor's floating-point unit.
- No dot-product, accumulator or pipeline controller is included.

## License

Copyright 2026 vera-rubin. Licensed under the Apache License 2.0, see [LICENSE](LICENSE).
