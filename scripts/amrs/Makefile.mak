# Makefile for preprocessing, building index, generating candidate pairs, and filtering with smatch

# Python interpreter
PYTHON = python3

# Script files
PREPROCESS_SCRIPT = preprocess.py
INDEX_BUILDER_SCRIPT = index_builder.py
CANDIDATE_PAIRS_SCRIPT = build_candidate_pairs.py
SMATCH_FILTER_SCRIPT = smatch_filter.py

# Default target
all: preprocess index build

# Preprocess step
preprocess:
	$(PYTHON) $(PREPROCESS_SCRIPT)

# Index builder step
index: preprocess
	$(PYTHON) $(INDEX_BUILDER_SCRIPT)

# Build candidate pairs step
build: index
	$(PYTHON) $(CANDIDATE_PAIRS_SCRIPT)

# Smatch filter step
smatch_filter: build
	$(PYTHON) $(SMATCH_FILTER_SCRIPT)

# Clean any temporary files (if necessary)
clean:
	rm -rf *.tmp
