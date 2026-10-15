"""Safe proc metadata contains exactly the owned namespace, never host escape links."""
from pathlib import Path

from atrex_runtime.workers.proc_view import snapshot


def process(root: Path, pid: int, parent: int, local: list[int], group: list[int],
            children: list[int], *, command: bytes = b'owned\0') -> None:
    directory = root / str(pid)
    (directory / 'task' / str(pid)).mkdir(parents=True, exist_ok=True)
    (directory / 'task' / str(pid) / 'children').write_text(' '.join(map(str, children)))
    (directory / 'stat').write_text(f'{pid} (my process) S {parent} {group[0]} {group[0]} '
                                   + ' '.join(['0'] * 18))
    (directory / 'status').write_text('Name:\tmy process\n' + '\n'.join(
        key + ':\t' + '\t'.join(map(str, values)) for key, values in
        [('NSpid', local), ('NSpgid', group), ('NSsid', group)]))
    (directory / 'cmdline').write_bytes(command)
    for forbidden in ['root', 'fd', 'environ', 'cwd', 'exe', 'mem']:
        (directory / forbidden).write_text('SECRET')


def test_proc_view_projects_local_identity_and_detached_children_without_host_files(tmp_path):
    proc, view = tmp_path / 'proc', tmp_path / 'view'
    view.mkdir()
    process(proc, 100, 50, [100], [100], [101])
    process(proc, 101, 100, [101, 1], [100, 0], [102])
    process(proc, 102, 101, [102, 2], [102, 2], [103])
    process(proc, 103, 102, [103, 3], [103, 3], [])  # detached session, still owned
    process(proc, 999, 50, [999, 1], [999, 1], [], command=b'UNRELATED_SECRET')
    assert snapshot(100, view, proc=proc)
    assert {p.name for p in view.iterdir()} == {'1', '2', '3', 'self'}
    assert (view / 'self').readlink() == Path('1')
    assert (view / '2/stat').read_text().startswith('2 (my process) S 1 2 2 ')
    assert (view / '2/task/2/children').read_text() == '3'
    assert 'NSpid:\t3' in (view / '3/status').read_text()
    for item in view.rglob('*'):
        assert item.name not in {'root', 'fd', 'cwd', 'exe', 'environ', 'mem'}
        if item.is_file():
            assert b'SECRET' not in item.read_bytes()
    process(proc, 102, 101, [102, 2], [102, 2], [])
    assert snapshot(100, view, proc=proc)
    assert not (view / '3').exists()


def test_proc_view_waits_for_child_namespace(tmp_path):
    proc, view = tmp_path / 'proc', tmp_path / 'view'
    view.mkdir()
    process(proc, 100, 50, [100], [100], [])
    assert snapshot(100, view, proc=proc) is False
    assert list(view.iterdir()) == []


def test_maps_broker_ancestry_rejects_peers_and_cycles(tmp_path):
    from atrex_runtime.workers.proc_view import owns_process
    process(tmp_path, 100, 50, [100], [100], [101])
    process(tmp_path, 101, 100, [101, 1], [100, 0], [102])
    process(tmp_path, 102, 101, [102, 2], [102, 2], [])
    process(tmp_path, 999, 50, [999, 1], [999, 1], [])
    assert owns_process(100, 102, proc=tmp_path)
    assert not owns_process(100, 999, proc=tmp_path)
    assert not owns_process(100, 50, proc=tmp_path)
    process(tmp_path, 999, 999, [999], [999], [])
    assert not owns_process(100, 999, proc=tmp_path)
