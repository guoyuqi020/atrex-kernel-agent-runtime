"""Bounded projections derived from immutable Session and Kernel artifacts."""

from __future__ import annotations

import base64
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore, StoredArtifact
from ..domain.ids import ArtifactDigest
from ..filesystem import regular_file_map
from ..workers.session_trace import retained_session_file

TRACE_PROJECTION_VERSION: Literal[2] = 2
KERNEL_DIFF_VERSION: Literal[1] = 1

_KNOWN_OMITTED_EVENTS = frozenset(
    {
        "turn/start",
        "step/start",
        "step/end",
        "assistant/chunk",
        "user/message",
        "request/header",
        "request/context",
        "session/end-seed",
        "todo/write",
        "agent/inbox/spliced",
        "command/run",
        "command/done",
        "approval/asked",
        "approval/decided",
        "approval/policy",
        "tool/code-dispatch-start",
        "tool/code-dispatch",
        "goal/change",
    }
)
_SELECTED_EVENTS = frozenset({"assistant/message", "tool/call", "tool/result", "turn/end"})
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password|authorization)"
    r"\b(\s*[:=]\s*)([^\s,;]+)"
)


@dataclass(frozen=True, slots=True)
class EvidenceProjectionLimits:
    """Deployment limits for derived evidence files."""

    max_trace_files: int
    max_trace_bytes: int
    max_trace_events: int
    max_projection_text_bytes: int
    max_diff_files: int
    max_diff_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.max_trace_files,
            self.max_trace_bytes,
            self.max_trace_events,
            self.max_projection_text_bytes,
            self.max_diff_files,
            self.max_diff_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Evidence projection limits must be positive")


class EvidenceArtifactProjector:
    """Read verified artifacts and emit bounded, non-authoritative derivatives."""

    def __init__(
        self,
        artifacts: LocalArtifactStore,
        limits: EvidenceProjectionLimits,
        *,
        redaction_patterns: tuple[str, ...] = (),
    ) -> None:
        self._artifacts = artifacts
        self._limits = limits
        self._redactions = tuple(re.compile(pattern) for pattern in redaction_patterns)

    def session_projection(self, digest: ArtifactDigest) -> dict[str, JsonValue]:
        """Project a bounded normalized summary while retaining only the source digest."""
        artifact, all_files = self._bounded_session(digest)
        files, compressed = self._semantic_session_files(artifact, all_files)
        if compressed:
            raise ValueError("Session projection requires uncompressed JSONL")
        if len(files) > self._limits.max_trace_files:
            raise ValueError("Session projection exceeds the configured file limit")

        sessions: list[JsonValue] = []
        remaining_events = self._limits.max_trace_events
        remaining_text = self._limits.max_projection_text_bytes
        for path in files:
            projected, used_events, used_text = self._project_session_file(
                path,
                remaining_events,
                remaining_text,
            )
            remaining_events -= used_events
            remaining_text -= used_text
            sessions.append(projected)
        return {
            "schema_version": TRACE_PROJECTION_VERSION,
            "source_session_log_digest": digest,
            "sessions": sessions,
        }

    def raw_session_projection(self, digest: ArtifactDigest) -> dict[str, JsonValue]:
        """Return every bounded retained Session file for authorized Evidence materialization."""
        artifact, all_files = self._bounded_session(digest)
        files: list[JsonValue] = []
        for path in all_files:
            relative = path.relative_to(artifact.payload_path).as_posix()
            payload, _removed = retained_session_file(relative, path.read_bytes())
            try:
                content = payload.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                content = base64.b64encode(payload).decode("ascii")
                encoding = "base64"
            files.append(
                {
                    "path": relative,
                    "encoding": encoding,
                    "content": content,
                }
            )
        return {
            "schema_version": 1,
            "source_session_log_digest": digest,
            "files": files,
        }

    def _bounded_session(self, digest: ArtifactDigest) -> tuple[StoredArtifact, list[Path]]:
        artifact = self._artifacts.verify(digest)
        if artifact.kind is not ArtifactKind.SESSION_LOG:
            raise ValueError("Session projection source has the wrong artifact kind")
        all_files = sorted(path for path in artifact.payload_path.rglob("*") if path.is_file())
        if sum(path.stat().st_size for path in all_files) > self._limits.max_trace_bytes:
            raise ValueError("Session trace exceeds the configured byte limit")
        return artifact, all_files

    @staticmethod
    def _semantic_session_files(
        artifact: StoredArtifact,
        all_files: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        """Separate Runtime event ledgers from opaque Provider transcripts.

        Provider-native JSONL, including Claude child-Agent transcripts, is retained in the
        immutable Session Artifact and in raw projections. It does not implement Runtime's
        normalized Session envelope and must never be parsed as one. ``conversation.jsonl`` is
        likewise a backend-neutral reading view rather than a semantic event ledger.
        """
        semantic: list[Path] = []
        compressed: list[Path] = []
        for path in all_files:
            relative = path.relative_to(artifact.payload_path)
            if not relative.parts or relative.parts[0] in {"provider", "input"}:
                continue
            if path.name == "conversation.jsonl":
                continue
            if path.name.endswith(".jsonl.zstd"):
                compressed.append(path)
            elif path.suffix == ".jsonl":
                semantic.append(path)
        return semantic, compressed

    def kernel_diff(
        self,
        before_digest: ArtifactDigest,
        after_digest: ArtifactDigest,
    ) -> dict[str, JsonValue]:
        """Create a deterministic bounded unified diff between verified Kernels."""
        before = self._artifacts.verify(before_digest)
        after = self._artifacts.verify(after_digest)
        if before.kind is not ArtifactKind.KERNEL or after.kind is not ArtifactKind.KERNEL:
            raise ValueError("Kernel diff sources must both be Kernel artifacts")
        before_files = self._regular_files(before.payload_path)
        after_files = self._regular_files(after.payload_path)
        paths = sorted(set(before_files).union(after_files))
        if len(paths) > self._limits.max_diff_files:
            raise ValueError("Kernel diff exceeds the configured file limit")

        changes: list[JsonValue] = []
        used_bytes = 0
        for relative in paths:
            old = before_files.get(relative)
            new = after_files.get(relative)
            old_bytes = b"" if old is None else old.read_bytes()
            new_bytes = b"" if new is None else new.read_bytes()
            if old_bytes == new_bytes:
                continue
            status = "added" if old is None else "deleted" if new is None else "modified"
            try:
                old_text = old_bytes.decode("utf-8")
                new_text = new_bytes.decode("utf-8")
            except UnicodeDecodeError:
                value: dict[str, JsonValue] = {
                    "path": relative,
                    "status": status,
                    "binary": True,
                    "before_bytes": len(old_bytes),
                    "after_bytes": len(new_bytes),
                }
            else:
                diff = "".join(
                    difflib.unified_diff(
                        old_text.splitlines(keepends=True),
                        new_text.splitlines(keepends=True),
                        fromfile=f"a/{relative}",
                        tofile=f"b/{relative}",
                        lineterm="\n",
                    )
                )
                diff = self._redact(diff)
                value = {
                    "path": relative,
                    "status": status,
                    "binary": False,
                    "unified_diff": diff,
                }
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            used_bytes += len(encoded)
            if used_bytes > self._limits.max_diff_bytes:
                raise ValueError("Kernel diff exceeds the configured byte limit")
            changes.append(value)
        return {
            "schema_version": KERNEL_DIFF_VERSION,
            "before_kernel_digest": before_digest,
            "after_kernel_digest": after_digest,
            "changes": changes,
        }

    def _project_session_file(
        self,
        path: Path,
        remaining_events: int,
        remaining_text: int,
    ) -> tuple[dict[str, JsonValue], int, int]:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("Session JSONL is empty")
        header = self._object(lines[0], "Session header")
        if (
            header.get("type") != "session"
            or header.get("version") != 0
            or not isinstance(header.get("id"), str)
        ):
            raise ValueError("Session JSONL has an invalid header")
        session_id = header["id"]
        if not isinstance(session_id, str):
            raise AssertionError("validated Session id is not a string")
        entries: list[JsonValue] = []
        previous_seq: int | None = None
        used_events = 0
        used_text = 0
        final_annotation: str | None = None
        for raw_line in lines[1:]:
            event = self._object(raw_line, "Session event")
            event_type = event.get("type")
            seq = event.get("seq")
            time = event.get("time")
            data = event.get("data")
            if not isinstance(event_type, str) or not isinstance(seq, int) or isinstance(seq, bool):
                raise ValueError("Session event identity is invalid")
            if not isinstance(time, int) or isinstance(time, bool) or not isinstance(data, dict):
                raise ValueError("Session event envelope is invalid")
            if previous_seq is not None and seq != previous_seq + 1:
                raise ValueError("Session event sequence is not contiguous")
            previous_seq = seq
            used_events += 1
            if used_events > remaining_events:
                raise ValueError("Session projection exceeds the configured event limit")
            if event_type in _KNOWN_OMITTED_EVENTS:
                continue
            if event_type not in _SELECTED_EVENTS:
                if event.get("ignorable") is True:
                    continue
                raise ValueError(
                    f"Session projection does not recognize required event {event_type!r}"
                )
            entry = self._project_event(event_type, seq, time, data)
            if entry is None:
                continue
            text_value = entry.get("text")
            if isinstance(text_value, str):
                encoded_size = len(text_value.encode())
                used_text += encoded_size
                if used_text > remaining_text:
                    raise ValueError("Session projection exceeds the configured text limit")
                if event_type == "assistant/message" and text_value.strip():
                    final_annotation = text_value
            entries.append(entry)
        relative = PurePosixPath(*path.parts[-3:]).as_posix()
        value: dict[str, JsonValue] = {
            "session_id": session_id,
            "source_file": relative,
            "entries": entries,
            "final_agent_annotation": final_annotation,
        }
        return value, used_events, used_text

    def _project_event(
        self,
        event_type: str,
        seq: int,
        time: int,
        data: dict[str, object],
    ) -> dict[str, JsonValue] | None:
        base: dict[str, JsonValue] = {"type": event_type, "seq": seq, "time": time}
        if event_type == "assistant/message":
            message = data.get("message")
            if not isinstance(message, dict):
                raise ValueError("assistant/message has no message object")
            text = self._content_text(message.get("content"))
            if not text:
                return None
            base["text"] = self._redact(text)
            return base
        if event_type == "tool/call":
            name = data.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("tool/call has an invalid name")
            base["tool_name"] = name
            return base
        if event_type == "tool/result":
            error = data.get("error")
            base["outcome"] = "error" if error is not None else "success"
            if error is not None:
                if not isinstance(error, dict):
                    raise ValueError("tool/result error is invalid")
                name = error.get("name")
                code = error.get("code")
                if not isinstance(name, str) or not isinstance(code, str):
                    raise ValueError("tool/result error identity is invalid")
                base["error_name"] = name
                base["error_code"] = code
            return base
        reason = data.get("reason")
        if not isinstance(reason, dict) or not isinstance(reason.get("kind"), str):
            raise ValueError("turn/end has an invalid reason")
        base["outcome"] = reason["kind"]
        return base

    @staticmethod
    def _content_text(value: object) -> str:
        if not isinstance(value, list):
            raise ValueError("assistant content must be a list")
        text: list[str] = []
        for block in value:
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                raise ValueError("assistant content block is invalid")
            if block["type"] == "text":
                part = block.get("text")
                if not isinstance(part, str):
                    raise ValueError("assistant text block is invalid")
                if part.strip():
                    text.append(part.strip())
        return "\n".join(text)

    def _redact(self, text: str) -> str:
        redacted = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
        for pattern in self._redactions:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    @staticmethod
    def _object(line: str, label: str) -> dict[str, object]:
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        return value

    @staticmethod
    def _regular_files(root: Path) -> dict[str, Path]:
        return regular_file_map(root)
