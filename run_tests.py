#!/usr/bin/env python3
"""One-command test for bf16_fp32_ops (``make test`` calls this). Standard library only.

Steps:
  1. Generate deterministic vectors (directed special-value grids, exhaustive sweeps over one BF16 operand,
     random and targeted random pairs: underflow, ties, cancellation, overflow, subnormals).
  2. Compute every expected word with the integer golden model and require the independent host-float
     oracle to agree (a golden/oracle disagreement fails the run before any RTL is involved).
  3. Compile the RTL and bench with Icarus Verilog (default) or Verilator, simulate, and require
     ``RESULT ... PASS`` with the expected vector counts.
  --mutants: apply each seeded bug (in a scratch copy, never in rtl/) and require the bench to FAIL.
  --lint:    ``verilator --lint-only -Wall`` on the RTL.

usage: python3 run_tests.py [--sim iverilog|verilator] [--quick] [--seed N] [--build DIR] [--mutants] [--lint]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import bf16_fp32_ref as ref  # noqa: E402

RTL = [HERE / "rtl" / f"{n}.sv" for n in ("a50_round_pack", "a50_bf16_mul", "a50_add32_align")]
TB = HERE / "tb" / "tb_bf16_fp32_ops.sv"
TOP = "tb_bf16_fp32_ops"

# Seeded bugs: (file, exact original text, replacement). Each must make the bench fail.
MUTANTS = {
    "ties_away": ("a50_round_pack.sv", "(rem == half && r[0])", "(rem == half)"),
    "no_overflow": ("a50_round_pack.sv", "if (quantum + 150 >= 255)", "if (quantum + 150 >= 256)"),
    "no_sticky": ("a50_add32_align.sv", "shifted | {26'd0, sticky}", "shifted"),
    "zero_sign_or": ("a50_add32_align.sv", "(a[31] & b[31])", "(a[31] | b[31])"),
    "inf_times_zero": ("a50_bf16_mul.sv", "(a_inf && b_zero) || (b_inf && a_zero)", "1'b0"),
    "flush_subnormal_in": ("a50_bf16_mul.sv", "{ea != 0, a[6:0]}", "{1'b1, a[6:0]}"),
}


def bf16_grid() -> list[int]:
    exps = (0, 1, 2, 63, 64, 126, 127, 128, 190, 191, 253, 254, 255)
    fracs = (0x00, 0x01, 0x40, 0x7F)
    return [(s << 15) | (e << 7) | f for s in (0, 1) for e in exps for f in fracs]


def f32_grid() -> list[int]:
    exps = (0, 1, 2, 23, 24, 25, 126, 127, 128, 150, 151, 253, 254, 255)
    fracs = (0, 1, 0x2AAAAA, 0x400000, 0x7FFFFF)
    return [(s << 31) | (e << 23) | f for s in (0, 1) for e in exps for f in fracs]


def mul_pairs(rng: random.Random, quick: bool) -> list[tuple[int, int]]:
    grid = bf16_grid()
    pairs = [(a, b) for a in grid for b in grid]
    # every BF16 value (incl. subnormals, infinities, NaNs) times a few fixed operands
    sweep_b = (0x3F80, 0x0001, 0x8080, 0x7F7F, 0x3FC0, 0x0D80) if not quick else (0x3F80, 0x0D80)
    pairs += [(a, b) for b in sweep_b for a in range(1 << 16)]
    n = 20_000 if quick else 100_000
    pairs += [(rng.getrandbits(16), rng.getrandbits(16)) for _ in range(n)]
    # products near and below the binary32 normal range: biased exponent sum ea + eb in [96, 132]
    for _ in range(n):
        ea = rng.randint(0, 132)
        eb = max(0, min(254, rng.randint(96, 132) - ea))
        pairs.append(((rng.getrandbits(1) << 15) | (ea << 7) | rng.getrandbits(7),
                      (rng.getrandbits(1) << 15) | (eb << 7) | rng.getrandbits(7)))
    return pairs


def add_pairs(rng: random.Random, quick: bool) -> list[tuple[int, int]]:
    grid = f32_grid()
    pairs = [(x, y) for x in grid for y in grid]
    n = 20_000 if quick else 100_000
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(n)]
    for _ in range(n):                       # near exponents: alignment, sticky, cancellation
        x = rng.getrandbits(32)
        e = (x >> 23) & 255
        d = rng.randint(0, 30)
        e2 = max(0, min(254, e - d if rng.random() < 0.5 else e + d))
        y = (rng.getrandbits(1) << 31) | (e2 << 23) | rng.getrandbits(23)
        if rng.random() < 0.1:
            y = x ^ 0x80000000               # exact cancellation, incl. -0 + +0
        pairs.append((x, y))
    for _ in range(n // 5):                  # exact ties: y is half an ulp of x (both lsb parities)
        e = rng.randint(25, 254)
        x = (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)
        y = (rng.getrandbits(1) << 31) | ((e - 24) << 23)
        pairs.append((x, y))
    for _ in range(n // 5):                  # near overflow
        x = (rng.getrandbits(1) << 31) | (rng.randint(252, 254) << 23) | rng.getrandbits(23)
        y = (rng.getrandbits(1) << 31) | (rng.randint(230, 254) << 23) | rng.getrandbits(23)
        pairs.append((x, y))
    for _ in range(n // 5):                  # subnormal + subnormal / small normal
        x = (rng.getrandbits(1) << 31) | (rng.randint(0, 2) << 23) | rng.getrandbits(23)
        y = (rng.getrandbits(1) << 31) | (rng.randint(0, 1) << 23) | rng.getrandbits(23)
        pairs.append((x, y))
    return pairs


def write_vectors(build: Path, seed: int, quick: bool) -> tuple[int, int]:
    rng = random.Random(seed)
    mul, add = mul_pairs(rng, quick), add_pairs(rng, quick)
    bad = 0
    with open(build / "mul.hex", "w") as f:
        for a, b in mul:
            p = ref.mul_bf16(a, b)
            if p != ref.oracle_mul_bf16(a, b):
                bad += 1
                print(f"golden/oracle MUL disagree {a:04x}*{b:04x}: {p:08x} vs {ref.oracle_mul_bf16(a, b):08x}")
            f.write(f"{a:04x} {b:04x} {p:08x}\n")
    with open(build / "add.hex", "w") as f:
        for x, y in add:
            s = ref.add32(x, y)
            if s != ref.oracle_add32(x, y):
                bad += 1
                print(f"golden/oracle ADD disagree {x:08x}+{y:08x}: {s:08x} vs {ref.oracle_add32(x, y):08x}")
            f.write(f"{x:08x} {y:08x} {s:08x}\n")
    print(f"vectors: mul={len(mul)} add={len(add)} golden_vs_oracle_mismatch={bad}")
    if bad:
        sys.exit("FAIL: golden model and host-float oracle disagree")
    return len(mul), len(add)


def simulate(build: Path, rtl: list[Path], sim: str) -> str:
    if sim == "iverilog":
        subprocess.run(["iverilog", "-g2012", "-Wall", "-Wno-timescale", "-o", str(build / "sim.vvp"), "-s", TOP,
                        *map(str, rtl), str(TB)], check=True)
        cmd = ["vvp", "-n", str(build / "sim.vvp")]
    else:
        obj = build / "obj_dir"
        subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                        "--top-module", TOP, "--Mdir", str(obj), "-o", "Vtb",
                        *map(str, rtl), str(TB)], check=True, stdout=subprocess.DEVNULL)
        cmd = [str(obj / "Vtb")]
    out = subprocess.run(cmd, cwd=build, check=True, capture_output=True, text=True).stdout
    return out


RESULT_RE = re.compile(r"RESULT mul=(\d+) mul_bad=(\d+) add=(\d+) add_bad=(\d+) (PASS|FAIL)")


def parse_result(out: str, n_mul: int, n_add: int) -> dict | None:
    """Return the bench's single, complete, self-consistent result frame, or None.

    None for: an ERROR line, no RESULT line or more than one, a RESULT line that is not exactly the
    expected format, populations that differ from the generated vector counts (so empty or partial
    runs never qualify), mismatch counts larger than their populations, or a verdict that contradicts
    the counts (PASS with mismatches, FAIL without any).
    """
    lines = out.splitlines()
    if any("ERROR" in line for line in lines):
        return None
    results = [line for line in lines if "RESULT" in line]
    if len(results) != 1:
        return None
    m = RESULT_RE.fullmatch(results[0].strip())
    if not m:
        return None
    r = {"mul": int(m[1]), "mul_bad": int(m[2]), "add": int(m[3]), "add_bad": int(m[4]), "verdict": m[5]}
    if r["mul"] != n_mul or r["add"] != n_add or n_mul <= 0 or n_add <= 0:
        return None
    if r["mul_bad"] > r["mul"] or r["add_bad"] > r["add"]:
        return None
    clean = r["mul_bad"] == 0 and r["add_bad"] == 0
    if (r["verdict"] == "PASS") != clean:
        return None
    return r


def passed(out: str, n_mul: int, n_add: int) -> bool:
    """Positive: complete populations, zero mismatches, PASS."""
    r = parse_result(out, n_mul, n_add)
    return r is not None and r["verdict"] == "PASS"


def killed(out: str, n_mul: int, n_add: int) -> bool:
    """Mutant detected: complete populations, FAIL, and at least one numerical mismatch.
    Missing files, missing or duplicate results, errors and empty runs are not kills."""
    r = parse_result(out, n_mul, n_add)
    return r is not None and r["verdict"] == "FAIL" and r["mul_bad"] + r["add_bad"] > 0


def parser_controls() -> bool:
    """Fixed bench transcripts that the result gate must accept or reject (no simulation involved)."""
    ok_line = "RESULT mul=2 mul_bad=0 add=3 add_bad=0 PASS"
    fail_line = "RESULT mul=2 mul_bad=1 add=3 add_bad=0 FAIL"
    cases = [  # (name, transcript, expected passed(), expected killed())
        ("genuine pass", ok_line, True, False),
        ("genuine kill", "MUL MISMATCH ...\n" + fail_line, False, True),
        ("pass with mul mismatches", "RESULT mul=2 mul_bad=1 add=3 add_bad=0 PASS", False, False),
        ("pass with add mismatches", "RESULT mul=2 mul_bad=0 add=3 add_bad=1 PASS", False, False),
        ("fail without mismatches", "RESULT mul=2 mul_bad=0 add=3 add_bad=0 FAIL", False, False),
        ("pass then fail", ok_line + "\n" + fail_line, False, False),
        ("fail then pass", fail_line + "\n" + ok_line, False, False),
        ("error then pass", "ERROR cannot open mul.hex\n" + ok_line, False, False),
        ("error, no result", "ERROR cannot open mul.hex", False, False),
        ("no output", "", False, False),
        ("empty populations", "RESULT mul=0 mul_bad=0 add=0 add_bad=0 FAIL", False, False),
        ("short mul population", "RESULT mul=1 mul_bad=0 add=3 add_bad=0 PASS", False, False),
        ("short add population, fail", "RESULT mul=2 mul_bad=1 add=2 add_bad=0 FAIL", False, False),
        ("mismatches exceed population", "RESULT mul=2 mul_bad=3 add=3 add_bad=0 FAIL", False, False),
        ("trailing junk", ok_line + " extra", False, False),
    ]
    good = True
    for name, text, want_pass, want_kill in cases:
        got = (passed(text, 2, 3), killed(text, 2, 3))
        if got != (want_pass, want_kill):
            good = False
            print(f"parser control FAILED: {name}: passed/killed={got}, expected {(want_pass, want_kill)}")
    print(f"parser controls: {len(cases)} cases, {'all as expected' if good else 'FAILURES'}")
    return good


def mutate(build: Path, name: str) -> list[Path]:
    fname, old, new = MUTANTS[name]
    mdir = build / f"mutant_{name}"
    mdir.mkdir(parents=True, exist_ok=True)
    out = []
    for src in RTL:
        text = src.read_text()
        if src.name == fname:
            if text.count(old) != 1:
                sys.exit(f"mutant {name}: pattern not found exactly once in {fname}")
            text = text.replace(old, new)
        dst = mdir / src.name
        dst.write_text(text)
        out.append(dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", choices=("iverilog", "verilator"), default="iverilog")
    ap.add_argument("--quick", action="store_true", help="smaller vector set")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--build", type=Path, default=HERE / "build")
    ap.add_argument("--mutants", action="store_true", help="require every seeded bug to fail the bench")
    ap.add_argument("--lint", action="store_true", help="verilator --lint-only -Wall on the RTL")
    args = ap.parse_args()
    build = args.build.resolve()
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    ok = parser_controls()
    if args.lint and shutil.which("verilator") is None:
        print("lint: FAILED (verilator not found on PATH; --lint requires it)")
        ok = False
    elif args.lint:
        r = subprocess.run(["verilator", "--lint-only", "-Wall", "--top-module", "a50_bf16_mul",
                            *map(str, RTL)], capture_output=True, text=True)
        r2 = subprocess.run(["verilator", "--lint-only", "-Wall", "--top-module", "a50_add32_align",
                             *map(str, RTL)], capture_output=True, text=True)
        lint_ok = r.returncode == 0 and r2.returncode == 0
        print("lint: verilator -Wall", "clean" if lint_ok else "FAILED\n" + r.stderr + r2.stderr)
        ok &= lint_ok
    n_mul, n_add = write_vectors(build, args.seed, args.quick)
    out = simulate(build, RTL, args.sim)
    shown = ("RESULT", "MISMATCH", "ERROR")
    print(f"rtl ({args.sim}): " + "\n".join(l for l in out.splitlines() if any(s in l for s in shown)))
    positive = passed(out, n_mul, n_add)
    if not positive:
        print("rtl: no single complete PASS frame with zero mismatches")
    ok &= positive
    if args.mutants:
        for name in MUTANTS:
            mout = simulate(build, mutate(build, name), args.sim)
            kill = killed(mout, n_mul, n_add)
            res = [l for l in mout.splitlines() if "RESULT" in l or "ERROR" in l]
            print(f"mutant {name}: {'killed' if kill else 'NOT KILLED'} ({' | '.join(res) or 'no RESULT'})")
            ok &= kill
    print("OVERALL", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
