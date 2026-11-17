# Reproduce everything:  make all   (about 5 minutes on a 10-core Apple M4; needs a C compiler with blocks, e.g. clang on macOS)
# Large outputs (results/cache, results/film, results/gpu/probe.jsonl) are not in git; they are regenerated here.
PY ?= python

all: dp/dp
	$(PY) run.py all

dp/dp: dp/dp.c
	clang -O2 -fblocks -o dp/dp dp/dp.c -lm

validate: dp/dp
	$(PY) run.py validate

gpucheck:
	$(PY) run.py gpucheck results

clean:
	rm -rf results/cache results/film dp/dp

.PHONY: all validate gpucheck clean
