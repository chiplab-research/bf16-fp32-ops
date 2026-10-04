# bf16_fp32_ops: one-command tests with free tools (Python 3 + Icarus Verilog; Verilator optional).
PYTHON ?= python3
SIM ?= iverilog
ARGS ?=

.PHONY: test quick mutants lint verilator clean

test:            ## full vector set, Icarus Verilog
	$(PYTHON) run_tests.py --sim $(SIM) $(ARGS)

quick:           ## smaller vector set
	$(PYTHON) run_tests.py --sim $(SIM) --quick $(ARGS)

mutants:         ## seeded bugs (applied in build/, never in rtl/) must each fail the bench
	$(PYTHON) run_tests.py --sim $(SIM) --quick --mutants $(ARGS)

lint:            ## verilator --lint-only -Wall
	$(PYTHON) run_tests.py --sim $(SIM) --quick --lint $(ARGS)

verilator:       ## full vector set, Verilator simulation
	$(PYTHON) run_tests.py --sim verilator $(ARGS)

clean:
	rm -rf build
