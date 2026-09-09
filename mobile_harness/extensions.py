"""Capability-scoped extension seam for native tools and future MCP clients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import hashlib
import json
from pathlib import Path
import subprocess
import threading

from .tools import ToolCall, ToolResult


@dataclass(frozen=True)
class Capability:
    name: str
    schema: dict[str, Any]
    category: str = "extension"
    version: str = "1.0"
    provenance: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class McpServerRecord:
    """A restart-safe server description confirmed by an explicit fingerprint."""
    server_name: str
    command: tuple[str, ...]
    allowlist: tuple[str, ...] | None
    fingerprint: str
    toolset_fingerprint: str = ""
    provenance: tuple[tuple[str, str], ...] = ()


@dataclass
class _LiveMcpServer:
    client: "StdioMcpClient"
    names: set[str]
    allowlist: set[str] | None
    record: McpServerRecord


class McpServerStore:
    """Persists server launch metadata, never auto-launching it by itself."""
    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "mcp_servers.json"

    @staticmethod
    def fingerprint(command: list[str] | tuple[str, ...], allowlist: set[str] | tuple[str, ...] | None,
                    provenance: tuple[tuple[str, str], ...] = ()) -> str:
        payload = json.dumps({"command": list(command), "allowlist": sorted(allowlist) if allowlist is not None else None,
                              "provenance": list(provenance)}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _legacy_fingerprint(command: list[str] | tuple[str, ...], allowlist: set[str] | tuple[str, ...] | None) -> str:
        payload = json.dumps({"command": list(command), "allowlist": sorted(allowlist) if allowlist is not None else None}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, server_name: str) -> McpServerRecord:
        rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        raw = rows.get(server_name)
        if not raw:
            raise KeyError(f"no persisted MCP server named {server_name}")
        provenance = tuple((str(key), str(value)) for key, value in raw.get("provenance", ()))
        record = McpServerRecord(raw["server_name"], tuple(raw["command"]), tuple(raw["allowlist"]) if raw.get("allowlist") is not None else None, raw["fingerprint"], raw.get("toolset_fingerprint", ""), provenance)
        actual = self.fingerprint(record.command, record.allowlist, record.provenance)
        if "provenance" not in raw:
            actual = self._legacy_fingerprint(record.command, record.allowlist)
        if actual != record.fingerprint:
            raise PermissionError("persisted MCP server record failed integrity validation")
        return record

    def save(self, record: McpServerRecord) -> None:
        rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        rows[record.server_name] = {"server_name": record.server_name, "command": list(record.command), "allowlist": list(record.allowlist) if record.allowlist is not None else None, "fingerprint": record.fingerprint, "toolset_fingerprint": record.toolset_fingerprint, "provenance": list(record.provenance)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


class CapabilityRegistry:
    """Explicit registration prevents plugins from silently widening authority."""
    def __init__(self) -> None:
        self._entries: dict[str, tuple[Capability, Callable[..., ToolResult], Callable[[], Any] | None]] = {}

    def register(self, capability: Capability, handler: Callable[..., ToolResult], *, availability: Callable[[], Any] | None = None) -> None:
        if capability.name in self._entries:
            raise ValueError(f"duplicate capability: {capability.name}")
        self._entries[capability.name] = (capability, handler, availability)

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def names(self) -> set[str]:
        return set(self._entries)

    def schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        return [entry[0].schema for name, entry in self._entries.items()
                if (allowed is None or name in allowed) and self.availability(name)["available"]]

    def compatible(self, name: str, minimum_version: str | None = None) -> dict[str, Any]:
        capability = self.capability(name)
        if not capability:
            return {"compatible": False, "reason": "not_registered", "version": ""}
        if not minimum_version:
            return {"compatible": True, "reason": "no_minimum", "version": capability.version}
        try:
            actual = tuple(int(part) for part in capability.version.split("."))
            minimum = tuple(int(part) for part in minimum_version.split("."))
        except ValueError:
            return {"compatible": False, "reason": "invalid_version", "version": capability.version}
        width = max(len(actual), len(minimum)); actual += (0,) * (width - len(actual)); minimum += (0,) * (width - len(minimum))
        return {"compatible": actual >= minimum, "reason": "compatible" if actual >= minimum else f"requires>={minimum_version}", "version": capability.version}

    def capability(self, name: str) -> Capability | None:
        entry = self._entries.get(name)
        return None if entry is None else entry[0]

    def availability(self, name: str) -> dict[str, Any]:
        """Probe an extension without granting it tool-call authority.

        A failing or malformed probe is fail-closed: the capability remains
        registered for diagnostics but is not advertised to the model.
        """
        entry = self._entries.get(name)
        if entry is None:
            return {"available": False, "reason": "not_registered"}
        probe = entry[2]
        if probe is None:
            return {"available": True, "reason": "no_probe"}
        try:
            value = probe()
            if isinstance(value, dict):
                return {"available": bool(value.get("available", True)), "reason": str(value.get("reason", "probe"))}
            return {"available": bool(value), "reason": "probe"}
        except Exception as exc:
            return {"available": False, "reason": f"probe_failed: {exc}"}

    def execute(self, call: ToolCall, *args: Any) -> ToolResult | None:
        entry = self._entries.get(call.name)
        if entry is None:
            return None
        status = self.availability(call.name)
        if not status["available"]:
            return ToolResult(False, error=f"extension tool unavailable: {status['reason']}")
        return entry[1](call.arguments, *args)


class McpToolAdapter:
    """Small adapter boundary: an MCP transport can supply callable schemas later.

    The adapter deliberately needs an explicit capability and handler for every
    remote tool, so a server cannot expose a new authority without registration.
    """
    def __init__(self, registry: CapabilityRegistry, store_root: str | Path | None = None) -> None:
        self.registry = registry
        self._clients: dict[str, _LiveMcpServer] = {}
        self.store = McpServerStore(store_root) if store_root else None

    def register_tool(self, name: str, parameters: dict[str, Any], invoke: Callable[[dict[str, Any]], Any], *, category: str = "extension") -> None:
        schema = {"type": "function", "function": {"name": name, "parameters": parameters}}
        self.registry.register(Capability(name, schema, category), lambda arguments, *_: ToolResult(True, invoke(arguments)))

    @staticmethod
    def _toolset_fingerprint(tools: list[dict[str, Any]], allowlist: set[str] | None = None) -> str:
        normalized = [{"name": str(tool.get("name", "")), "inputSchema": tool.get("inputSchema", {"type": "object"})}
                      for tool in tools if tool.get("name") and (allowlist is None or str(tool.get("name")) in allowlist)]
        return hashlib.sha256(json.dumps(sorted(normalized, key=lambda item: item["name"]), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def register_stdio_server(self, command: list[str], *, allowlist: set[str] | None = None) -> list[str]:
        """Discover MCP-style JSON schemas using a bounded `tools/list` request.

        Transport is intentionally one-shot; tool invocation still requires explicit
        registration and therefore cannot inherit arbitrary server authority.
        """
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
        completed = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=15, check=True)
        response = json.loads(next(line for line in completed.stdout.splitlines() if line.strip()))
        registered = []
        for tool in response.get("result", {}).get("tools", []):
            name = str(tool.get("name", ""))
            if not name or (allowlist is not None and name not in allowlist): continue
            schema = tool.get("inputSchema", {"type": "object"})
            self.register_tool(name, schema, lambda arguments, n=name: {"server": command[0], "tool": n, "arguments": arguments})
            registered.append(name)
        return registered

    def connect_stdio_server(self, command: list[str], *, allowlist: set[str] | None = None, server_name: str = "mcp",
                             package_provenance: dict[str, str] | None = None) -> list[str]:
        """Keep a JSON-RPC process alive for discovery and subsequent invocations."""
        if server_name in self._clients:
            raise ValueError(f"MCP server already connected: {server_name}")
        client = StdioMcpClient(command)
        try:
            tools = client.request("tools/list", {}).get("tools", [])
        except Exception:
            client.close()
            raise
        provenance = tuple(sorted((str(key), str(value)) for key, value in (package_provenance or {}).items()))
        record = McpServerRecord(server_name, tuple(command), tuple(sorted(allowlist)) if allowlist is not None else None, McpServerStore.fingerprint(command, allowlist, provenance), self._toolset_fingerprint(tools, allowlist), provenance)
        registered = []
        for tool in tools:
            name = str(tool.get("name", ""))
            if not name or (allowlist is not None and name not in allowlist):
                continue
            schema = tool.get("inputSchema", {"type": "object"})
            full_name = f"mcp__{server_name}__{name}"
            self.registry.register(Capability(full_name, {"type": "function", "function": {"name": full_name, "parameters": schema}}, "mcp", provenance=provenance), lambda arguments, *_, n=name, c=client: ToolResult(True, c.request("tools/call", {"name": n, "arguments": arguments})))
            registered.append(full_name)
        self._clients[server_name] = _LiveMcpServer(client, set(registered), allowlist, record)
        if self.store:
            self.store.save(record)
        return registered

    def restore_trusted(self, server_name: str, expected_fingerprint: str) -> list[str]:
        """Reconnect only when the caller confirms the stored server identity."""
        if not self.store:
            raise RuntimeError("MCP persistence is not configured")
        record = self.store.load(server_name)
        if record.fingerprint != expected_fingerprint:
            raise PermissionError("MCP server fingerprint did not match explicit trust confirmation")
        return self.connect_stdio_server(list(record.command), allowlist=set(record.allowlist) if record.allowlist is not None else None, server_name=record.server_name, package_provenance=dict(record.provenance))

    def server_record(self, server_name: str) -> McpServerRecord:
        return self._clients[server_name].record

    def refresh(self, server_name: str) -> list[str]:
        """Refresh a live server's advertised schemas without retaining stale tools."""
        live = self._clients[server_name]
        for name in live.names: self.registry.unregister(name)
        tools = live.client.request("tools/list", {}).get("tools", [])
        names = set()
        for tool in tools:
            short = str(tool.get("name", ""))
            if not short or (live.allowlist is not None and short not in live.allowlist): continue
            full = f"mcp__{server_name}__{short}"; schema = tool.get("inputSchema", {"type": "object"})
            self.registry.register(Capability(full, {"type": "function", "function": {"name": full, "parameters": schema}}, "mcp", provenance=live.record.provenance), lambda arguments, *_, n=short, c=live.client: ToolResult(True, c.request("tools/call", {"name": n, "arguments": arguments})))
            names.add(full)
        live.names = names
        updated = McpServerRecord(live.record.server_name, live.record.command, live.record.allowlist, live.record.fingerprint, self._toolset_fingerprint(tools, live.allowlist), live.record.provenance)
        live.record = updated
        if self.store: self.store.save(updated)
        return sorted(names)

    def refresh_provenance(self, server_name: str) -> dict[str, str]:
        """Refresh schemas and return the auditable toolset-contract transition."""
        before = self._clients[server_name].record.toolset_fingerprint
        self.refresh(server_name)
        return {"before": before, "after": self._clients[server_name].record.toolset_fingerprint}

    def disconnect(self, server_name: str) -> None:
        live = self._clients.pop(server_name)
        for name in live.names: self.registry.unregister(name)
        live.client.close()

    def close(self) -> None:
        for server_name in list(self._clients): self.disconnect(server_name)


class StdioMcpClient:
    """Minimal persistent newline-delimited JSON-RPC client for MCP stdio servers."""
    def __init__(self, command: list[str], request_timeout: float = 30.0, cancel_grace: float = 0.25) -> None:
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._id, self._lock = 0, threading.Lock()
        self.request_timeout, self.cancel_grace = request_timeout, cancel_grace

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.process.stdin or not self.process.stdout: raise RuntimeError("MCP transport is unavailable")
        with self._lock:
            self._id += 1; request_id = self._id
            self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n"); self.process.stdin.flush()
            while True:
                line = self._readline_with_timeout(request_id)
                if not line: raise RuntimeError("MCP server exited before responding")
                response = json.loads(line)
                if response.get("id") != request_id: continue
                if "error" in response: raise RuntimeError(str(response["error"]))
                return response.get("result", {})

    def _readline_with_timeout(self, request_id: int) -> str:
        """Bound a read, requesting MCP cancellation before the hard-stop fallback."""
        if not self.process.stdout: raise RuntimeError("MCP transport is unavailable")
        result: list[str] = []
        done = threading.Event()
        def read() -> None:
            try: result.append(self.process.stdout.readline())
            finally: done.set()
        threading.Thread(target=read, daemon=True).start()
        if not done.wait(self.request_timeout):
            self._notify_cancel(request_id)
            # Servers that honor notifications/cancelled can wind down the request
            # without losing their transport. The outstanding reader drains a late
            # response before the next request, which will ignore its old id.
            if not done.wait(self.cancel_grace):
                self.cancel()
            raise TimeoutError(f"MCP request exceeded {self.request_timeout:g}s")
        return result[0] if result else ""

    def _notify_cancel(self, request_id: int) -> None:
        if self.process.poll() is not None or not self.process.stdin:
            return
        try:
            self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": request_id, "reason": "client request timeout"}}) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            # The hard-stop fallback below handles a dead or non-cooperating peer.
            return

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                stream.close()

    def cancel(self) -> None:
        """Transport-level cancellation: terminates the active server process."""
        self.close()
