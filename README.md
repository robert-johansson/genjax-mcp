# GenJAX MCP Server

An [MCP](https://modelcontextprotocol.io/) server that gives Claude Code (or any MCP client) expert-level knowledge of the [GenJAX](https://github.com/femtomc/genjax) probabilistic programming library.

**Self-contained** — all documentation, examples, source code, and tests are bundled in `data/`. No GenJAX checkout required for documentation and search. Optionally point `GENJAX_ROOT` at a live checkout to enable code execution.

## What's included

**95 files** bundled across 4 kinds:

| Kind | Count | Description |
|------|-------|-------------|
| **source** | 19 | Library implementation (`core.py`, `pjax.py`, `distributions.py`, `inference/smc.py`, etc.) |
| **learn** | 24 | Small standalone examples (generative functions, combinators, choice maps, inference) |
| **case_study** | 31 | Full case studies — localization, curve fitting, Game of Life, MCTS, state space models, etc. |
| **test** | 21 | Complete test suite |

Plus **19 documentation files** covering core API, inference algorithms, ADEV, Gaussian processes, and each case study.

## Tools

| Tool | Purpose |
|------|---------|
| `get_documentation(topic)` | Retrieve documentation by topic |
| `search_examples(query, category)` | Keyword search across all 95 files, filterable by category |
| `get_example(name)` | Fetch a specific file by name |
| `validate_genjax_code(code)` | Execute code against a GenJAX environment (requires `GENJAX_ROOT`) |

## Setup

```bash
git clone https://github.com/femtomc/genjax-mcp.git
cd genjax-mcp
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Configuration

### Claude Code

Add to `.mcp.json` in your project root or `~/.claude/mcp.json` for global access:

```json
{
  "mcpServers": {
    "genjax": {
      "command": "/absolute/path/to/genjax-mcp/venv/bin/python",
      "args": ["/absolute/path/to/genjax-mcp/server.py"]
    }
  }
}
```

Paths **must be absolute**.

### Enabling code execution

The `validate_genjax_code` tool runs code against a real GenJAX environment. To enable it, set `GENJAX_ROOT` pointing at a GenJAX checkout with a working Python environment (`venv/` or pixi):

```json
{
  "mcpServers": {
    "genjax": {
      "command": "/absolute/path/to/genjax-mcp/venv/bin/python",
      "args": ["/absolute/path/to/genjax-mcp/server.py"],
      "env": {
        "GENJAX_ROOT": "/path/to/your/genjax"
      }
    }
  }
}
```

Without `GENJAX_ROOT`, the other 3 tools (docs, search, get) work normally — only `validate_genjax_code` is disabled.

### Setting up a GenJAX environment for code execution

GenJAX is **not on PyPI** (the `genjax` package on PyPI is an older, unrelated version). Install from the GitHub repo instead.

**Option A: pixi (recommended)**

The repo uses [pixi](https://pixi.sh) for environment management. If you have pixi installed:

```bash
git clone https://github.com/femtomc/genjax.git
cd genjax
pixi install
```

This handles JAX, CUDA, and all dependencies automatically. Then set `GENJAX_ROOT` to this directory. The server will use `pixi run python` to execute code.

**Option B: pip + venv (tested on macOS arm64, Python 3.13)**

Requires Python 3.12+.

```bash
git clone https://github.com/femtomc/genjax.git
cd genjax
python3 -m venv venv
source venv/bin/activate

# Install JAX first (CPU-only; see https://jax.readthedocs.io for GPU)
pip install jax==0.6.2 jaxlib==0.6.2

# Install GenJAX in editable mode (pulls in beartype, tensorflow-probability, etc.)
pip install -e .

# matplotlib is required at import time but not listed in dependencies
pip install matplotlib
```

Verify it works:

```bash
python -c "from genjax import gen, normal; print('OK')"
```

Then point `GENJAX_ROOT` at this directory in your `.mcp.json`. The server will use `venv/bin/python` to execute code.

**Known issues:**
- `matplotlib` is imported by `genjax.viz` at startup but isn't in `pyproject.toml` dependencies (it's provided by pixi/conda instead). You must install it manually with pip.
- The PyPI package `genjax` (versions 0.9.x–0.10.x) is from a different, older project — **do not** `pip install genjax`. Always install from the GitHub repo with `pip install -e .`.

### Other MCP clients

The server uses stdio transport. Run directly:

```bash
venv/bin/python server.py
```

## Usage examples

**"How does SMC work in GenJAX?"**
→ `get_documentation("inference")`

**"Show me how to use Scan"**
→ `search_examples("Scan", category="combinators")`

**"What's the implementation of importance sampling?"**
→ `search_examples("importance_sampling", category="source")`

**"Get the full SMC source"**
→ `get_example("src/inference/smc")`

**"Get the SMC test file"**
→ `get_example("tests/test_smc")`

### Name conventions for `get_example`

| Kind | Format | Examples |
|------|--------|----------|
| Learn | `{name}` | `gf_scan`, `beta_bernoulli_smc`, `choice_maps_build_merge` |
| Case study | `{case}/{file}` | `localization/core`, `curvefit/main`, `gol/data` |
| Source | `src/{path}` | `src/core`, `src/inference/smc`, `src/gp/kernels` |
| Test | `tests/{name}` | `tests/test_mcmc`, `tests/test_smc`, `tests/conftest` |

### Searchable categories

`adev`, `case_studies`, `choice_maps`, `combinators`, `core`, `extras`, `gp`, `hmc`, `inference`, `mcmc`, `mcts`, `particle_filter`, `smc`, `source`, `tests`, `vi`, `viz`

### Documentation topics

`root`, `core`, `inference`, `adev`, `extras`, `gp`, `viz`, `tests`, `examples`, `simple_intro`, `api_reference`, `curvefit`, `faircoin`, `gen2d`, `gol`, `intuitive_physics`, `localization`, `programmable_mcts`, `state_space`

## Updating bundled data

To refresh the bundled files from a GenJAX checkout:

```bash
# From a GenJAX repo with all examples present
GENJAX=/path/to/genjax

# Docs
for f in CLAUDE.md src/genjax/CLAUDE.md src/genjax/inference/CLAUDE.md \
  src/genjax/adev/CLAUDE.md src/genjax/extras/CLAUDE.md src/genjax/gp/CLAUDE.md \
  src/genjax/viz/CLAUDE.md tests/CLAUDE.md examples/CLAUDE.md \
  examples/simple_intro/CLAUDE.md examples/simple_intro/genjax_current_api.md \
  examples/curvefit/CLAUDE.md examples/faircoin/CLAUDE.md examples/gen2d/CLAUDE.md \
  examples/gol/CLAUDE.md examples/intuitive_physics/CLAUDE.md \
  examples/localization/CLAUDE.md examples/programmable_mcts/CLAUDE.md \
  examples/state_space/CLAUDE.md; do
  flat=$(echo "$f" | sed 's|/|--|g')
  cp "$GENJAX/$f" data/docs/"$flat"
done

# Learn examples
cp "$GENJAX"/examples/learn/examples/*.py data/learn/

# Case studies (core.py, main.py, data.py — skip figs/export)
for dir in "$GENJAX"/examples/*/; do
  name=$(basename "$dir")
  [ "$name" = "learn" ] || [ "$name" = "__pycache__" ] && continue
  mkdir -p data/case_studies/"$name"
  for py in "$dir"*.py; do
    stem=$(basename "$py" .py)
    case "$stem" in figs|export|visualizations|__init__) continue ;; esac
    cp "$py" data/case_studies/"$name"/
  done
done
cp "$GENJAX"/examples/*.py data/case_studies/ 2>/dev/null

# Source
rsync -a --include='*.py' --include='*/' --exclude='*' \
  --exclude='__pycache__' "$GENJAX"/src/genjax/ data/src/

# Tests
cp "$GENJAX"/tests/*.py data/tests/
rm -f data/tests/__init__.py
```

## File structure

```
genjax-mcp/
├── server.py              # MCP server
├── requirements.txt       # Python dependencies (mcp[cli])
├── README.md
└── data/
    ├── docs/              # 19 CLAUDE.md and API reference files
    ├── learn/             # 24 learn-by-example snippets
    ├── case_studies/      # 31 case study files
    │   ├── localization/
    │   ├── curvefit/
    │   ├── ...
    │   ├── utils.py       # shared utilities
    │   └── viz.py
    ├── src/               # 19 library source files
    │   ├── core.py
    │   ├── pjax.py
    │   ├── inference/
    │   │   ├── smc.py
    │   │   ├── mcmc.py
    │   │   └── vi.py
    │   └── ...
    └── tests/             # 21 test files
```
