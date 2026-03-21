# Quick syntax check (indentation, brackets, etc.) without running the app.
PYTHON ?= python3

.PHONY: check

check:
	$(PYTHON) -m py_compile app.py rag_engine.py prompts.py
	@echo "OK — all Python files compile."
