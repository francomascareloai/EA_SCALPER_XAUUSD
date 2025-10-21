## Installation with uv

```bash
# Create python environment
uv venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

# Install project and dependencies
uv pip install -e .
# To include development dependencies:
uv pip install -e ".[dev]"
```

## Installation with pip

```bash
# Create python environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate

# Install project and dependencies
pip install -e .
# To include development dependencies:
pip install -e ".[dev]"
```
