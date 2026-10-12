"""Guard the compact CSA input against size and input-semantics regressions."""

import ast
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "ppubench"
ADAPTED = (
    DATA
    / "agate-inputs"
    / "compressed_sparse_attention_paged_causal_bf16_interleaved_kv_append"
)


class _Normalize(ast.NodeTransformer):
    """Undo only the documented edits; retain all executable expressions."""

    def __init__(self, renames, removed_constants):
        self.renames = renames
        self.removed_constants = removed_constants

    def visit_Module(self, node):
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        node.body = [
            child
            for child in node.body
            if not (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and child.targets[0].id in self.removed_constants
            )
        ]
        return self.generic_visit(node)

    def visit_Name(self, node):
        node.id = self.renames.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.renames.get(node.arg, node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = self.renames.get(node.name, node.name)
        node.returns = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        assert node.value is not None
        return self.generic_visit(ast.Assign(targets=[node.target], value=node.value))


def test_csa_compact_input_preserves_executable_ast_and_fits_gateway_limit():
    manifest = json.loads((ADAPTED / "adaptation.json").read_text())
    original = (DATA / manifest["upstream_input"]).read_bytes()
    compact = (ADAPTED / "input.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == manifest["upstream_sha256"]
    assert hashlib.sha256(compact).hexdigest() == manifest["compact_sha256"]
    assert len(original) == manifest["upstream_bytes"]
    assert len(compact) == manifest["compact_bytes"] <= manifest["limit_bytes"] == 8192

    before, after = ast.parse(original.decode("utf-8")), ast.parse(compact.decode("utf-8"))
    compile(after, "compact-input.py", "exec")
    entries = [
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_make_inputs")
        for tree in (before, after)
    ]
    assert ast.dump(entries[0].args) == ast.dump(entries[1].args)
    assert ast.dump(entries[0].returns) == ast.dump(entries[1].returns)

    renames = manifest["renames"]
    # Renamed bindings must be fresh, so shortening cannot capture another variable.
    names = {
        n.id if isinstance(n, ast.Name) else n.arg
        for n in ast.walk(before)
        if isinstance(n, (ast.Name, ast.arg))
    }
    names.update(n.name for n in before.body if isinstance(n, ast.FunctionDef))
    assert len(set(renames.values())) == len(renames)
    assert not names.intersection(renames.values())
    removed = set(manifest["removed_unused_constants"])
    assert not any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in removed
        for n in ast.walk(before)
    )
    normalized_before = _Normalize({}, removed).visit(before)
    normalized_after = _Normalize({v: k for k, v in renames.items()}, removed).visit(after)
    assert ast.dump(normalized_before) == ast.dump(normalized_after)
