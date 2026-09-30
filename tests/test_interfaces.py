"""Keep the CLI, MCP server and SciLink plug-in in sync with each other.

MCP signatures are read with ``ast`` so these checks run without the ``mcp``
package; a separate test imports the real server when ``mcp`` is installed.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest

from mlmd_exafs.calculators import BACKENDS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tool"))

import mlmd_exafs_tool  # noqa: E402

STAGES = {
    "mlmd_relax", "mlmd_md", "mlmd_feff_input", "mlmd_run_feff", "mlmd_cleanup",
    "mlmd_average_chi", "mlmd_fit_e0", "mlmd_lcf", "mlmd_plot", "mlmd_convergence",
}

# Output-location params differ by design: SciLink writes into the session's
# output_dir and takes a savefile name, MCP takes an explicit path.
OUTPUT_PARAM_DIFFS = {
    "mlmd_relax": ({"structure_path"}, {"output"}),
    "mlmd_md": (set(), {"directory"}),
    "mlmd_fit_e0": ({"savefile"}, {"outdir"}),
    "mlmd_lcf": ({"savefile"}, {"outdir"}),
}

SCHEMAS = {s["function"]["name"]: s["function"] for s in mlmd_exafs_tool.tool_schemas}
TOOL_FUNCS = mlmd_exafs_tool.create_tool_functions("structure.cif", "out")


def _mcp_signatures():
    tree = ast.parse((REPO / "mlmd_exafs" / "mcp_server.py").read_text())
    sigs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "tool"
            for d in node.decorator_list
        ):
            args = node.args
            n_required = len(args.args) - len(args.defaults)
            sigs[node.name] = {
                "all": {a.arg for a in args.args},
                "required": {a.arg for a in args.args[:n_required]},
            }
    return sigs


MCP = _mcp_signatures()


def test_all_layers_expose_the_same_stages():
    assert set(SCHEMAS) == STAGES
    assert set(TOOL_FUNCS) == STAGES
    assert set(MCP) == STAGES


@pytest.mark.parametrize("name", sorted(STAGES))
def test_scilink_schema_matches_function(name):
    params = SCHEMAS[name]["parameters"]
    sig = inspect.signature(TOOL_FUNCS[name]).parameters
    assert set(params["properties"]) == set(sig)
    for pname, p in sig.items():
        is_required = p.default is inspect.Parameter.empty
        assert (pname in params.get("required", [])) == is_required, pname


@pytest.mark.parametrize("name", sorted(STAGES))
def test_scilink_and_mcp_params_agree(name):
    schema_only, mcp_only = OUTPUT_PARAM_DIFFS.get(name, (set(), set()))
    schema_params = set(SCHEMAS[name]["parameters"]["properties"])
    mcp_params = MCP[name]["all"]
    assert schema_params - mcp_params <= schema_only
    assert mcp_params - schema_params == mcp_only


def test_backend_enum_matches_backends():
    for name in ("mlmd_relax", "mlmd_md"):
        assert SCHEMAS[name]["parameters"]["properties"]["backend"]["enum"] == list(BACKENDS)


def test_cli_backend_args_cover_mcp_backend_params():
    from mlmd_exafs.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    for cmd in ("relax", "md"):
        dests = {a.dest for a in sub.choices[cmd]._actions}
        assert {"backend", "device", "model", "checkpoint", "head", "modal"} <= dests


def test_mcp_server_imports_and_registers_tools():
    pytest.importorskip("mcp")
    from mlmd_exafs import mcp_server

    for name in STAGES:
        assert callable(getattr(mcp_server, name))


def test_scilink_cleanup_and_average(exafs_dir, tmp_path):
    funcs = mlmd_exafs_tool.create_tool_functions("structure.cif", str(tmp_path))
    avg = funcs["mlmd_average_chi"](str(exafs_dir), savefile="s")
    assert avg["status"] == "success"
    assert avg["n_samples"] == 4
    assert "chi_avg" not in avg  # arrays stripped for JSON
    assert Path(avg["output_file"]) == tmp_path / "s-chi_avg.dat"

    clean = funcs["mlmd_cleanup"](str(exafs_dir))
    assert clean["status"] == "success"
    assert clean["deleted_files"] == 8

    empty = tmp_path / "empty"
    empty.mkdir()
    assert funcs["mlmd_cleanup"](str(empty))["status"] == "no_data"
