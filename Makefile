PYTHON ?= python

.PHONY: build
build:
	$(PYTHON) tools/package_addon.py
