# Docker GPU workflow. Host needs only Docker + the NVIDIA Container Toolkit.
IMAGE := cuda-motion-flow:dev
ROOT  := $(shell pwd)
MOUNTS := -v $(ROOT):/workspace -v $(ROOT)/data:/workspace/data \
          -v $(ROOT)/weights:/workspace/weights -v $(ROOT)/outputs:/workspace/outputs
RUN := docker run --rm --gpus all $(MOUNTS) -w /workspace $(IMAGE)

.PHONY: build smoke test shell

build:
	docker build -t $(IMAGE) $(ROOT)

smoke:
	$(RUN) python3 scripts/smoke_test.py

test:
	$(RUN) python3 -m pytest

shell:
	docker run --rm --gpus all $(MOUNTS) -w /workspace -it $(IMAGE) bash
