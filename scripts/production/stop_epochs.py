"""Reconcile a managed workspace's Epochs after its processes have exited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atrex_runtime.config import RuntimeStorageSettings
from atrex_runtime.domain.ids import LineageId, parse_campaign_id, parse_lineage_id
from atrex_runtime.domain.models import EpochStatus
from atrex_runtime.registry.sqlite import SqliteRegistry


def stop_workspace_epochs(workspace: Path) -> list[str]:
    """Target only Lineages recorded by this workspace's Bootstrap/arm seed results.

    The shell caller must verify that all workspace processes have exited first.
    No services, credentials, GPU operations or Agent imports are needed here.
    """
    workspace = workspace.resolve(strict=True)
    config = json.loads((workspace / "runtime.json").read_text())
    storage = RuntimeStorageSettings.model_validate(config["storage"]).resolve_from(workspace)
    if not storage.registry_database.is_file():
        return []  # Stopped before Bootstrap ever opened the Registry.
    result_paths: list[Path] = []
    for dsl in ("cuda", "triton", "cutedsl"):
        root = workspace / "dsls" / dsl
        result_paths.append(root / "bootstrap-result.json")
        result_paths.extend(sorted(root.glob("*/seed-result.json")))
    with SqliteRegistry(storage.registry_database) as registry:
        lineages: set[LineageId] = set()
        for path in result_paths:
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(workspace):
                raise ValueError(f"Campaign result escapes workspace: {path}")
            value = json.loads(path.read_text())
            campaign_id = parse_campaign_id(value["campaign_id"])
            entries = value["lineages"] if "lineages" in value else [value["lineage"]]
            for entry in entries:
                lineage_id = parse_lineage_id(entry["lineage_id"])
                if registry.get_lineage(lineage_id).campaign_id != campaign_id:
                    raise ValueError(f"Campaign/Lineage mismatch in {path}")
                lineages.add(lineage_id)
        stopped: list[str] = []
        for lineage_id in sorted(lineages):
            epoch = registry.find_open_epoch(lineage_id)
            if epoch is None:
                continue
            result = registry.stop_epoch(epoch.id, "managed campaign stopped by operator")
            if result.status is EpochStatus.STOPPED:
                stopped.append(str(result.id))
        return stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    stopped = stop_workspace_epochs(args.workspace)
    print(json.dumps({"stopped_epochs": stopped}, sort_keys=True))


if __name__ == "__main__":
    main()
