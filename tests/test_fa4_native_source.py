"""Regress native Eval's whole-tree gate and R0's compatibility-only source patch."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/FA4"
BASE = "7b077cf391a98a06ad9464530f0a4c9c7be3f477"
CUTE = "vendor/flash_attention/flash_attn/cute/"
QUACK = "vendor_support/quack/"


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    target = tmp_path_factory.mktemp("fa4-native-source") / "source"
    subprocess.run(["git", "clone", "--quiet", str(DATA / "source.bundle"), str(target)],
                   check=True, capture_output=True)
    return target


def original(source: Path, path: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), "show", f"{BASE}:{path}"],
                                   text=True)


def namespace(text: str, names: set[str], values: dict) -> dict:
    tree = ast.parse(text)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                                          names=[ast.alias(name="annotations")]), *nodes],
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), "<source-test>", "exec"), values)
    return values


def test_entire_fa4_source_passes_pinned_agate_gate(source: Path) -> None:
    fixture = Path(__file__).parent / "fixtures/agate_source_gate"
    provenance = json.loads((fixture / "provenance.json").read_text())
    env = {}
    for component in provenance["files"][::-1]:
        path = fixture / (Path(component["path"]).name + ".txt")
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == component["sha256"]
        tree = ast.parse(data)
        # Load the two unmodified upstream modules into one isolated test namespace.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "app.security.files")]
        exec(compile(tree, str(path), "exec"), env)
    files = {p.relative_to(source).as_posix(): p.read_text() for p in source.rglob("*.py")}
    files["kernel.py"] = (DATA / "task/adapter.py").read_text()
    assert len(files) == 65
    all_files = {p.relative_to(source).as_posix(): p.read_text() for p in source.rglob("*")
                 if p.is_file() and ".git" not in p.parts}
    all_files["kernel.py"] = files["kernel.py"]
    assert env["validate_sources"](all_files, "kernel.py::Model") == []
    bad_metadata = {**all_files, "PROVENANCE.json": '{"note": "environment override"}'}
    assert any("PROVENANCE.json:" in e for e in env["validate_sources"](
        bad_metadata, "kernel.py::Model"))
    errors = {name: issues for name, text in files.items() if (
        issues := env["validate_kernel_source"](text, require_model_class=name == "kernel.py"))}
    assert errors == {}
    assert env["validate_kernel_source"]("import sys\n", require_model_class=False)


class OriginalImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module:
            node.module = node.module.replace("vendor.flash_attention.flash_attn", "flash_attn")
            node.module = node.module.replace("vendor_support.quack", "quack")
        return node

    def visit_Import(self, node):
        for alias in node.names:
            alias.name = alias.name.replace("vendor.flash_attention.flash_attn", "flash_attn")
            if alias.name == "vendor_support.quack.activation":
                alias.name, alias.asname = "quack.activation", None
        return node

    def visit_Name(self, node):
        if node.id == "quack_activation":
            return ast.Attribute(value=ast.Name(id="quack", ctx=ast.Load()),
                                 attr="activation", ctx=node.ctx)
        return node


def test_algorithm_and_dispatch_are_original_not_the_optimized_bootstrap(source: Path) -> None:
    for name in ["sm100_hd256_2cta_fmha_forward.py", "flash_bwd_sm100.py",
                 "sm100_hd256_2cta_fmha_backward_dkdvkernel.py"]:
        old = ast.parse(original(source, CUTE + name))
        new = OriginalImports().visit(ast.parse((source / CUTE / name).read_text()))
        assert ast.dump(old) == ast.dump(new), name
    old = ast.parse(original(source, CUTE + "interface.py"))
    new = OriginalImports().visit(ast.parse((source / CUTE / "interface.py").read_text()))
    for name in ["_flash_attn_fwd", "_flash_attn_bwd", "flash_attn_varlen_func"]:
        left = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        right = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(left) == ast.dump(right), name


def test_compatibility_patch_reproduces_every_bundled_file(source: Path, tmp_path: Path) -> None:
    checkout = tmp_path / "source"
    subprocess.run(["git", "clone", "--quiet", str(DATA / "source.bundle"), str(checkout)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", BASE], check=True)
    subprocess.run(["git", "-C", str(checkout), "apply", str(DATA / "patches/native-eval.patch")],
                   check=True)
    integrity = json.loads((DATA / "asset-integrity.json").read_text())["components"]["source"]
    files = {p.relative_to(checkout).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in checkout.rglob("*") if p.is_file() and ".git" not in p.parts}
    assert files == integrity["files_sha256"]


def test_process_local_jit_cache_retains_separate_keys_and_instances(source: Path) -> None:
    env = namespace((source / CUTE / "cache_utils.py").read_text(),
                    {"JITCache", "get_jit_cache"}, {})
    cache = env["get_jit_cache"]("fwd")
    keys = [("dtype", 128), ("dtype", 256), ("dtype", 128, "different_callback")]
    for index, key in enumerate(keys):
        cache[key] = index
    for index, key in enumerate(keys):
        assert key in cache and cache[key] == index
    assert keys[0] not in env["get_jit_cache"]("bwd")
    cache.clear()
    assert all(key not in cache for key in keys)


def test_callback_metadata_and_hash_keep_jit_specialization(source: Path) -> None:
    protocols = {}
    exec((source / CUTE / "_metadata_protocols.py").read_text(), protocols)
    names = {"_compute_base_hash", "hash_callable"}
    common = {"hashlib": hashlib, "inspect": inspect, "Callable": Callable,
              "_MIXER_ATTRS": ("__vec_size__",)}
    old = namespace(original(source, CUTE + "utils.py"), names, dict(common))
    new = namespace((source / CUTE / "utils.py").read_text(), names, {**common, **protocols})

    def callback():
        def fn(x):
            return x + 1
        return fn

    for vec in (None, 0, 1, 2, 4):
        for precomputed in (False, True):
            a, b = callback(), callback()
            a.__vec_size__ = b.__vec_size__ = vec
            if precomputed:
                a.__cute_hash__ = b.__cute_hash__ = "precomputed"
            assert old["hash_callable"](a) == new["hash_callable"](b)
    for obj in (SimpleNamespace(), SimpleNamespace(__assumed_align__=16, __leading_dim__=1)):
        assert protocols["aux_tensor_layout"](obj, 0, -1) == (
            getattr(obj, "__assumed_align__", 0), getattr(obj, "__leading_dim__", -1),
            hasattr(obj, "__leading_dim__"),
        )


def test_explicit_params_protocol_preserves_field_order_and_fails_on_omissions(source: Path):
    common = {"fields": fields, "StaticTypes": (int, bool, str, float, type(None))}
    text = (source / QUACK / "cute_dsl_utils.py").read_text()
    old = namespace(original(source, QUACK + "cute_dsl_utils.py"),
                    {"_partition_fields"}, dict(common))
    new = namespace(text, {"_partition_fields"}, dict(common))

    @dataclass
    class Params:
        tile: int
        tensor: object
        optional: object = None

        def __quack_field_items__(self):
            return (("tile", self.tile), ("tensor", self.tensor), ("optional", self.optional))

    params = Params(128, object())
    assert old["_partition_fields"](params) == new["_partition_fields"](params)

    @dataclass
    class Extended(Params):
        extra: object = None

    with pytest.raises(ValueError, match="declaration order"):
        new["_partition_fields"](Extended(128, object()))


def test_namedtuple_registration_preserves_converter_type(source: Path) -> None:
    common = {"get_origin": lambda _: None, "cutlass": SimpleNamespace(Constexpr=object()),
              "_original_convert_single_arg": lambda a, n, t, c: t}
    names = {"_patched_convert_single_arg"}
    old = namespace(original(source, QUACK + "cute_dsl_utils.py"), names, dict(common))
    new = namespace((source / QUACK / "cute_dsl_utils.py").read_text(),
                    names | {"register_namedtuple_type"}, {**common, "_NAMEDTUPLE_TYPES": []})

    class Args(NamedTuple):
        value: object

    new["register_namedtuple_type"](Args)
    for value in (Args(1), (1,), None):
        for typ in (None, tuple, Args):
            assert old["_patched_convert_single_arg"](value, "a", typ, None) == (
                new["_patched_convert_single_arg"](value, "a", typ, None))


def test_copy_operation_protocol_handles_wrappers_and_missing_pack(source: Path) -> None:
    env = namespace((source / QUACK / "copy_utils.py").read_text(),
                    {"_copy_operation", "_load_pack"}, {})
    bare = SimpleNamespace(repeat=4)
    wrapped = SimpleNamespace(op=bare)
    for obj in (bare, wrapped, SimpleNamespace(op=wrapped)):
        old = obj.op if hasattr(obj, "op") else obj
        old = old.op if hasattr(old, "op") else old
        assert env["_copy_operation"](env["_copy_operation"](obj)) is old
    assert env["_load_pack"](bare) is None
    bare.pack = "packed"
    assert env["_load_pack"](bare) == "packed"
