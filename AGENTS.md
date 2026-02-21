# AGENTS.md

This file provides guidance for AI coding agents operating in this repository.

## Repository Overview

This is a machine learning workspace focused on LLMs (Large Language Models). The primary
codebase is `LLMs-from-scratch/`, a companion repo for "Build a Large Language Model (From
Scratch)" by Sebastian Raschka. It contains chapter-by-chapter implementations of GPT-2,
Llama 3, Qwen3, Gemma 3, and OLMo 3, along with an installable Python package (`llms-from-scratch`).

The root also has a `docker-compose.yaml` for running JupyterLab (with .NET support) and
infrastructure services (Postgres, pgAdmin, Redis) via Docker Compose.

## Build & Run Commands

### Environment Setup

```bash
# Install core dependencies (from LLMs-from-scratch/)
pip install -r requirements.txt

# Install the package in editable mode (from LLMs-from-scratch/)
pip install -e ".[dev,bonus]"

# Alternative: use pixi (conda-based) environment
pixi install
```

### Docker

```bash
# Start JupyterLab + infrastructure (from repo root)
docker compose up -d infrastructure

# JupyterLab is at http://localhost:8080
```

### Running Tests

```bash
# Run all tests (from LLMs-from-scratch/)
pytest

# Run a single test file
pytest pkg/llms_from_scratch/tests/test_ch04.py

# Run a single test function
pytest pkg/llms_from_scratch/tests/test_ch04.py::test_gpt_model

# Run a specific parametrized test
pytest pkg/llms_from_scratch/tests/test_ch04.py::test_gpt_model[GPTModel] -v

# Run chapter-level tests (standalone test scripts)
pytest ch04/01_main-chapter-code/tests.py
pytest ch05/07_gpt_to_llama/tests/tests_rope_and_parts.py

# Run with verbose output
pytest -v -s pkg/llms_from_scratch/tests/

# Run tests matching a keyword
pytest -k "llama" -v
```

### Linting

```bash
# Run ruff linter (configuration in pyproject.toml)
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Spell checking
codespell
```

### Notebook Validation

```bash
# Validate Jupyter notebooks with nbval
pytest --nbval ch02/01_main-chapter-code/ch02.ipynb
```

## Code Style Guidelines

### Formatter & Linter: Ruff

Configuration is in `LLMs-from-scratch/pyproject.toml`:

- **Line length**: 140 characters max
- **Suppressed rules**: E402 (late imports OK), E722 (bare except OK), E731 (lambda assignment OK),
  E741 (ambiguous variable names OK), E226, E702, E703, C406

### File Headers

Every Python file **must** begin with the 4-line copyright block:

```python
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
```

### Import Order

Follow PEP 8 loosely — standard library first, then third-party, then local:

```python
import os
import json

import torch
import torch.nn as nn
import tiktoken
import numpy as np

from .ch03 import MultiHeadAttention
from ..generate import trim_input_tensor
```

Late imports (after code) are permitted (E402 is suppressed). Relative imports are used
within the `pkg/llms_from_scratch/` package.

### Naming Conventions

| Element       | Convention         | Examples                                           |
|---------------|--------------------|----------------------------------------------------|
| Classes       | `PascalCase`       | `GPTModel`, `TransformerBlock`, `Qwen3Tokenizer`   |
| Functions     | `snake_case`       | `generate_text_simple`, `compute_rope_params`       |
| Constants     | `UPPER_SNAKE_CASE` | `GPT_CONFIG_124M`, `LLAMA32_CONFIG_1B`              |
| Variables     | `snake_case`       | `attn_scores`, `context_vec`, `num_heads`           |
| Private       | `_leading_under`   | `_extract_imports`, `_header`, `_hf_ids`            |
| Math vars     | Short names OK     | `x`, `b`, `q`, `k`, `v`, `d_out`                   |

### Type Annotations

Type annotations are **minimal** throughout. This is an educational codebase that prioritizes
readability over static typing. Do not add type hints unless they significantly aid clarity
(e.g., constructor signatures like `tokenizer: Llama3Tokenizer`).

### Docstrings & Comments

- **Inline comments are heavily used** — this is educational code. Comment significant operations,
  especially tensor shape transformations (e.g., `# Shape: (b, num_tokens, d_out)`).
- **Class docstrings**: Include when a class has a non-obvious variant or architectural difference
  (e.g., `GPTModelFast` documents how it differs from `GPTModel`).
- **Method docstrings**: Generally omitted; use inline comments instead.
- **Section dividers**: Use `####` comment blocks to visually separate logical sections.

### Error Handling

- Use **assertions** for preconditions: `assert d_out % num_heads == 0, "d_out must be divisible by num_heads"`
- Use **explicit exceptions** with descriptive messages:
  `raise ValueError(f"Shape mismatch in tensor '{tensor_name}'...")`
- Bare `except` clauses are acceptable (E722 suppressed) but prefer specific exceptions when possible.
- For I/O operations, raise `FileNotFoundError` or `RuntimeError` with context.

### Architecture Patterns

- **Config-dict pattern**: Model configurations are plain Python dicts (e.g., `GPT_CONFIG_124M = {...}`).
- **nn.Module inheritance**: All models and layers follow standard PyTorch module patterns.
- **Variant pattern**: Base class + "Fast" variant using optimized ops (FlashAttention,
  `scaled_dot_product_attention`). Example: `GPTModel` / `GPTModelFast`.
- **KV-cache variants**: Separate subpackages (`kv_cache/`, `kv_cache_batched/`) that mirror base structure.

### Test Patterns

- Use **pytest** with fixtures (`@pytest.fixture`, `@pytest.fixture(scope="session")`).
- Use **`@pytest.mark.parametrize`** for testing multiple model/function variants.
- Set **`torch.manual_seed`** for reproducibility in every test.
- Use **`torch.testing.assert_close`** or **`torch.equal`** for tensor comparison.
- Hardcode **deterministic reference values** for regression testing.
- Use **`@pytest.mark.skipif`** for optional-dependency tests (e.g., `transformers` not installed).

### Project Structure

```
LLMs-from-scratch/
  pkg/llms_from_scratch/       # Installable package (PyPI: llms-from-scratch)
    ch02.py - ch07.py          # Chapter module code
    llama3.py, qwen3.py        # Full model implementations
    generate.py, utils.py      # Shared utilities
    kv_cache/                  # KV-cache optimized variants
    kv_cache_batched/          # Batched KV-cache variants
    tests/                     # Package-level pytest tests
  ch01/ - ch07/                # Chapter directories with notebooks and bonus content
  appendix-A/ - appendix-E/   # Appendix materials
  conftest.py                  # Root pytest config (link-checking)
  pyproject.toml               # Build config, dependencies, ruff settings
```

### CI/CD

GitHub Actions runs on every push/PR to `main`:
- **Tests**: Cross-platform (Linux, macOS, Windows) with multiple Python versions and package managers
- **Linting**: Ruff (PEP 8 style)
- **Spell check**: codespell
- **Link check**: Validates URLs in docs and notebooks

### Key Dependencies

- **PyTorch** (>=2.2.2) — core ML framework
- **tiktoken** — OpenAI tokenizer
- **matplotlib** — visualization
- **huggingface_hub**, **safetensors**, **transformers** (>=5.0.0) — model weights
- **pytest**, **pytest-ruff**, **nbval** — testing
