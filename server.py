"""GenJAX MCP Server — gives Claude Code expert-level GenJAX knowledge.

Self-contained: reads from bundled data/ directory by default.
Set GENJAX_ROOT to override with a live GenJAX checkout.

Tools: get_documentation, search_examples, get_example, validate_genjax_code
Resources: genjax://docs/{topic}, genjax://examples/{name}
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MCP_DIR = Path(__file__).resolve().parent
DATA_DIR = MCP_DIR / "data"

# Optional: override with a live GenJAX checkout for validate_genjax_code
GENJAX_ROOT = Path(os.environ["GENJAX_ROOT"]) if "GENJAX_ROOT" in os.environ else None

# ---------------------------------------------------------------------------
# Documentation: topic → flat filename in data/docs/
# ---------------------------------------------------------------------------

DOCS_MAP: dict[str, str] = {
    "root": "CLAUDE.md",
    "core": "src--genjax--CLAUDE.md",
    "inference": "src--genjax--inference--CLAUDE.md",
    "adev": "src--genjax--adev--CLAUDE.md",
    "extras": "src--genjax--extras--CLAUDE.md",
    "gp": "src--genjax--gp--CLAUDE.md",
    "viz": "src--genjax--viz--CLAUDE.md",
    "tests": "tests--CLAUDE.md",
    "examples": "examples--CLAUDE.md",
    "simple_intro": "examples--simple_intro--CLAUDE.md",
    "api_reference": "examples--simple_intro--genjax_current_api.md",
    "curvefit": "examples--curvefit--CLAUDE.md",
    "faircoin": "examples--faircoin--CLAUDE.md",
    "gen2d": "examples--gen2d--CLAUDE.md",
    "gol": "examples--gol--CLAUDE.md",
    "intuitive_physics": "examples--intuitive_physics--CLAUDE.md",
    "localization": "examples--localization--CLAUDE.md",
    "programmable_mcts": "examples--programmable_mcts--CLAUDE.md",
    "state_space": "examples--state_space--CLAUDE.md",
}

# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

LEARN_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "core": ["gf_", "simulate_", "tuple_args", "random_keys"],
    "inference": ["beta_bernoulli_", "importance_", "gf_native_importance", "update_trace"],
    "combinators": ["gf_scan", "gf_repeat", "gf_vmap", "conditional_", "branching_gaussian"],
    "choice_maps": ["choice_maps_"],
}

CASE_STUDY_CATEGORIES: dict[str, list[str]] = {
    "curvefit": ["inference", "mcmc", "hmc"],
    "faircoin": ["inference", "core"],
    "gen2d": ["inference", "core"],
    "gol": ["inference", "smc"],
    "intuitive_physics": ["inference", "smc"],
    "localization": ["inference", "smc", "particle_filter"],
    "programmable_mcts": ["inference", "mcts"],
    "simple_intro": ["core"],
    "state_space": ["inference", "smc"],
}

TEST_CATEGORIES: dict[str, list[str]] = {
    "test_core": ["core", "tests"],
    "test_distributions": ["core", "tests"],
    "test_pjax": ["core", "tests"],
    "test_state": ["core", "tests"],
    "test_mcmc": ["inference", "mcmc", "tests"],
    "test_smc": ["inference", "smc", "tests"],
    "test_vi": ["inference", "vi", "tests"],
    "test_adev": ["adev", "tests"],
    "test_sp": ["core", "tests"],
    "test_gp": ["gp", "tests"],
    "test_gp_invariants": ["gp", "tests"],
    "test_linear_gaussian": ["inference", "tests"],
    "test_discrete_hmm": ["inference", "smc", "tests"],
    "test_vmap_rejuvenation_smc": ["inference", "smc", "tests"],
    "test_vmap_generate_bug": ["core", "tests"],
    "test_mvnormal_estimators": ["inference", "tests"],
    "test_benchmarks": ["core", "tests"],
    "test_simple_benchmark": ["core", "tests"],
    "test_utils": ["core", "tests"],
    "conftest": ["tests"],
    "conftest_jit": ["tests"],
}

SOURCE_CATEGORIES: dict[str, list[str]] = {
    "core": ["source", "core"],
    "distributions": ["source", "core"],
    "pjax": ["source", "core"],
    "sp": ["source", "core"],
    "state": ["source", "core"],
    "__init__": ["source", "core"],
    "adev/__init__": ["source", "adev"],
    "inference/__init__": ["source", "inference"],
    "inference/mcmc": ["source", "inference", "mcmc"],
    "inference/smc": ["source", "inference", "smc"],
    "inference/vi": ["source", "inference", "vi"],
    "extras/__init__": ["source", "extras"],
    "extras/state_space": ["source", "extras", "smc"],
    "gp/__init__": ["source", "gp"],
    "gp/gp": ["source", "gp"],
    "gp/kernels": ["source", "gp"],
    "gp/mean": ["source", "gp"],
    "viz/__init__": ["source", "viz"],
    "viz/raincloud": ["source", "viz"],
}

SKIP_STEMS = {"figs", "export", "visualizations", "__init__"}


def _classify_learn_example(name: str) -> list[str]:
    cats = []
    for cat, patterns in LEARN_CATEGORY_PATTERNS.items():
        if any(name.startswith(p) or p in name for p in patterns):
            cats.append(cat)
    return cats or ["core"]


# ---------------------------------------------------------------------------
# Build unified index from bundled data/
# ---------------------------------------------------------------------------


def _load_all_examples() -> dict[str, dict]:
    """Load all indexed files from the bundled data/ directory."""
    examples: dict[str, dict] = {}

    # 1. Learn examples
    learn_dir = DATA_DIR / "learn"
    if learn_dir.is_dir():
        for path in sorted(learn_dir.glob("*.py")):
            name = path.stem
            examples[name] = {
                "name": name,
                "path": str(path),
                "source": path.read_text(),
                "categories": _classify_learn_example(name),
                "kind": "learn",
            }

    # 2. Case study files
    cs_dir = DATA_DIR / "case_studies"
    if cs_dir.is_dir():
        # Subdirectory case studies
        for case_dir in sorted(cs_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_name = case_dir.name
            base_cats = CASE_STUDY_CATEGORIES.get(case_name, ["core"])
            for py_file in sorted(case_dir.glob("*.py")):
                indexed_name = f"{case_name}/{py_file.stem}"
                examples[indexed_name] = {
                    "name": indexed_name,
                    "path": str(py_file),
                    "source": py_file.read_text(),
                    "categories": ["case_studies"] + base_cats,
                    "kind": "case_study",
                }

        # Top-level example files
        for py_file in sorted(cs_dir.glob("*.py")):
            name = f"examples/{py_file.stem}"
            examples[name] = {
                "name": name,
                "path": str(py_file),
                "source": py_file.read_text(),
                "categories": ["core"],
                "kind": "case_study",
            }

    # 3. Library source code
    src_dir = DATA_DIR / "src"
    if src_dir.is_dir():
        for py_file in sorted(src_dir.rglob("*.py")):
            rel = py_file.relative_to(src_dir).with_suffix("")
            rel_str = str(rel).replace(os.sep, "/")
            cats = SOURCE_CATEGORIES.get(rel_str, ["source"])
            indexed_name = f"src/{rel_str}"
            examples[indexed_name] = {
                "name": indexed_name,
                "path": str(py_file),
                "source": py_file.read_text(),
                "categories": cats,
                "kind": "source",
            }

    # 4. Test files
    test_dir = DATA_DIR / "tests"
    if test_dir.is_dir():
        for py_file in sorted(test_dir.glob("*.py")):
            indexed_name = f"tests/{py_file.stem}"
            cats = TEST_CATEGORIES.get(py_file.stem, ["tests"])
            examples[indexed_name] = {
                "name": indexed_name,
                "path": str(py_file),
                "source": py_file.read_text(),
                "categories": cats,
                "kind": "test",
            }

    return examples


EXAMPLES = _load_all_examples()
ALL_CATEGORIES = sorted({cat for ex in EXAMPLES.values() for cat in ex["categories"]})

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("genjax-assistant")


@mcp.tool()
def get_documentation(topic: str) -> str:
    """Get GenJAX documentation for a topic.

    Available topics: root, core, inference, adev, extras, gp, viz, tests,
    examples, simple_intro, api_reference, curvefit, faircoin, gen2d, gol,
    intuitive_physics, localization, programmable_mcts, state_space
    """
    topic = topic.strip().lower()
    if topic not in DOCS_MAP:
        available = ", ".join(sorted(DOCS_MAP.keys()))
        return f"Unknown topic '{topic}'. Available topics: {available}"
    doc_file = DATA_DIR / "docs" / DOCS_MAP[topic]
    if not doc_file.exists():
        return f"Documentation file not found: {doc_file}"
    return doc_file.read_text()


@mcp.tool()
def search_examples(query: str, category: str = "all") -> str:
    """Search GenJAX learn examples by keyword.

    Args:
        query: Search term (case-insensitive substring match).
        category: Filter by category — core, inference, combinators, choice_maps, or all.

    Returns top 5 matches with full source code.
    """
    query_lower = query.lower()
    category = category.strip().lower()

    candidates = list(EXAMPLES.values())
    if category != "all":
        candidates = [e for e in candidates if category in e["categories"]]

    scored: list[tuple[float, dict]] = []
    for ex in candidates:
        searchable = (ex["name"] + " " + ex["source"]).lower()
        count = searchable.count(query_lower)
        if count > 0:
            boost = 1.2 if ex["kind"] == "learn" else 1.0
            scored.append((count * boost, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    if not top:
        cat_hint = f" in category '{category}'" if category != "all" else ""
        avail = ", ".join(ALL_CATEGORIES)
        return f"No examples found matching '{query}'{cat_hint}.\nAvailable categories: {avail}"

    parts: list[str] = [f"Found {len(top)} matching example(s) for '{query}':\n"]
    for score, ex in top:
        cats = ", ".join(ex["categories"])
        parts.append(f"### {ex['name']} ({ex['kind']}, matches: {score:.0f}, categories: {cats})")
        parts.append(f"```python\n{ex['source']}```\n")

    return "\n".join(parts)


@mcp.tool()
def get_example(name: str) -> str:
    """Get a specific GenJAX learn example by name.

    Args:
        name: Example name (e.g., 'gf_scan', 'beta_bernoulli_smc').
              For source files use 'src/core', 'src/inference/smc', etc.
              For tests use 'tests/test_mcmc', etc.
              Omit the .py extension.
    """
    name = name.strip().removesuffix(".py")
    if name in EXAMPLES:
        ex = EXAMPLES[name]
        return f"# {ex['name']}\n# Kind: {ex['kind']}\n# Categories: {', '.join(ex['categories'])}\n\n{ex['source']}"

    available = ", ".join(sorted(EXAMPLES.keys()))
    return f"Unknown example '{name}'. Available examples: {available}"


@mcp.tool()
def validate_genjax_code(code: str) -> str:
    """Validate GenJAX code by executing it.

    Requires GENJAX_ROOT to be set (pointing at a GenJAX checkout with
    a working Python environment). Returns stdout, stderr, and exit code.
    Timeout: 30 seconds.
    """
    if GENJAX_ROOT is None:
        return (
            "Error: GENJAX_ROOT is not set. To use validate_genjax_code, set the "
            "GENJAX_ROOT environment variable to point at a GenJAX checkout with "
            "a working Python environment (venv/ or pixi)."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    genjax_venv_python = GENJAX_ROOT / "venv" / "bin" / "python"

    try:
        if genjax_venv_python.exists():
            result = subprocess.run(
                [str(genjax_venv_python), tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(GENJAX_ROOT),
            )
        else:
            result = subprocess.run(
                ["pixi", "run", "python", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(GENJAX_ROOT),
            )
    except FileNotFoundError:
        python_candidates = [
            GENJAX_ROOT / ".pixi" / "envs" / "default" / "bin" / "python",
            "python3",
        ]
        executed = False
        result = None
        for py in python_candidates:
            try:
                result = subprocess.run(
                    [str(py), tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(GENJAX_ROOT),
                )
                executed = True
                break
            except (FileNotFoundError, OSError):
                continue
        if not executed or result is None:
            os.unlink(tmp_path)
            return "Error: Could not find a suitable Python interpreter."
    except subprocess.TimeoutExpired:
        os.unlink(tmp_path)
        return "Error: Code execution timed out after 30 seconds."
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    assert result is not None
    parts = []
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    parts.append(f"Exit code: {result.returncode}")

    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    return f"[{status}]\n" + "\n".join(parts)


# --- Resources ---


def _make_doc_reader(filename: str):
    def reader() -> str:
        path = DATA_DIR / "docs" / filename
        if path.exists():
            return path.read_text()
        return f"File not found: {path}"
    return reader


def _make_example_reader(ex: dict):
    def reader() -> str:
        return ex["source"]
    return reader


for _topic, _filename in DOCS_MAP.items():
    mcp.resource(f"genjax://docs/{_topic}", name=f"docs-{_topic}")(_make_doc_reader(_filename))

for _name, _ex in EXAMPLES.items():
    _uri_name = _name.replace("/", "--")
    mcp.resource(f"genjax://examples/{_uri_name}", name=f"example-{_uri_name}")(_make_example_reader(_ex))


if __name__ == "__main__":
    mcp.run()
