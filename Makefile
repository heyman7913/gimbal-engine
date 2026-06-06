# Docker GPU workflow. Host needs only Docker + the NVIDIA Container Toolkit.
IMAGE := gimbal:dev
ROOT  := $(shell pwd)
MOUNTS := -v $(ROOT):/workspace -v $(ROOT)/data:/workspace/data \
          -v $(ROOT)/weights:/workspace/weights -v $(ROOT)/outputs:/workspace/outputs
RUN := docker run --rm --gpus all $(MOUNTS) -w /workspace $(IMAGE)

.PHONY: image ext smoke test shell

image:
	docker build -t $(IMAGE) $(ROOT)

# compile the cuda extension into src/ (needed before test/cli)
ext:
	$(RUN) python3 setup.py build_ext --inplace

smoke:
	$(RUN) python3 scripts/smoke_test.py

test:
	$(RUN) bash -lc "python3 setup.py build_ext --inplace && python3 -m pytest"

shell:
	docker run --rm --gpus all $(MOUNTS) -w /workspace -it $(IMAGE) bash
