# LLM

A machine learning workspace focused on Large Language Models.

## Contents

- **Notebooks/** - Jupyter notebooks for hands-on experimentation
- **SharpGPT/** - GPT implementation in C#/.NET
- **LLMs-from-scratch/** - Companion code for *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka

## Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with drivers installed (for GPU-accelerated workloads)

### Running JupyterLab

```bash
docker compose up -d infrastructure
```

JupyterLab will be available at [http://localhost:8080](http://localhost:8080) (token: `docker`).

### Python Environment (for LLMs-from-scratch)

```bash
cd LLMs-from-scratch
pip install -r requirements.txt
pip install -e ".[dev,bonus]"
```

### Running Tests

```bash
cd LLMs-from-scratch
pytest
```
