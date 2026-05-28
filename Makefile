CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -std=c11 -mavx2 -mfma -Isrc
LDFLAGS = -lm

SRCDIR  = src
UTILDIR = utils
BINDIR  = bin
TMPDIR  = tmp

# Default test parameters
N       ?= 4096
DENSITY ?= 0.5
SEED    ?= 42
ARGS    = $(N) $(DENSITY) $(SEED)

# Source groups
SRC_COMMON   = $(UTILDIR)/common.c
SRC_BASELINE = $(SRCDIR)/baseline.c
SRC_STUDENT  = $(SRCDIR)/optimized.c

.PHONY: all clean correctness test

all: $(BINDIR)/nbody_baseline $(BINDIR)/nbody_optimized \
     $(BINDIR)/nbody_verify $(BINDIR)/nbody_correctness

# --- Directories ---

$(BINDIR) $(TMPDIR):
	mkdir -p $@

# --- Binaries ---

$(BINDIR)/nbody_baseline: $(UTILDIR)/run_baseline.c $(SRC_BASELINE) $(SRC_COMMON) $(SRCDIR)/nbody.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(UTILDIR)/run_baseline.c $(SRC_BASELINE) $(SRC_COMMON) $(LDFLAGS)

$(BINDIR)/nbody_optimized: $(UTILDIR)/run_optimized.c $(SRC_BASELINE) $(SRC_STUDENT) $(SRC_COMMON) $(SRCDIR)/nbody.h | $(BINDIR)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(UTILDIR)/run_optimized.c $(SRC_BASELINE) $(SRC_STUDENT) $(SRC_COMMON) $(LDFLAGS)

$(BINDIR)/nbody_verify: $(UTILDIR)/verify.c $(SRC_COMMON) $(SRCDIR)/nbody.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(UTILDIR)/verify.c $(SRC_COMMON) $(LDFLAGS)

$(BINDIR)/nbody_correctness: $(UTILDIR)/bruteforce.c $(SRC_BASELINE) $(SRC_COMMON) $(SRCDIR)/nbody.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $(UTILDIR)/bruteforce.c $(SRC_BASELINE) $(SRC_COMMON) $(LDFLAGS)

# --- Targets ---

correctness: $(BINDIR)/nbody_correctness
	$(BINDIR)/nbody_correctness 512 0.5 42
	@echo "---"
	$(BINDIR)/nbody_correctness 1000 0.3 123
	@echo "---"
	$(BINDIR)/nbody_correctness 2000 0.8 999

test: $(BINDIR)/nbody_baseline $(BINDIR)/nbody_optimized $(BINDIR)/nbody_verify | $(TMPDIR)
	$(BINDIR)/nbody_baseline $(ARGS) $(TMPDIR)/forces_baseline.bin 2>/dev/null
	$(BINDIR)/nbody_optimized $(ARGS) $(TMPDIR)/forces_optimized.bin 2>/dev/null
	$(BINDIR)/nbody_verify $(TMPDIR)/forces_baseline.bin $(TMPDIR)/forces_optimized.bin

clean:
	-$(MAKE) -C $(SRCDIR) clean
	rm -rf $(BINDIR) $(TMPDIR)
