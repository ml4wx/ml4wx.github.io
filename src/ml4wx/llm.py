"""Run Claude through the Claude Code CLI (`claude -p`).

The pipeline runs on a Claude subscription token (CLAUDE_CODE_OAUTH_TOKEN), which works
only through Claude Code, not the API SDK. Each call replaces the default system prompt,
disables tools, settings and MCP servers, and asks for structured output.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def structured(prompt: str, system: str, schema: dict, model: str, timeout: int = 600) -> dict:
    """One `claude -p` call; returns the object that matches `schema`."""
    exe = shutil.which("claude")
    if not exe:
        raise LLMError("claude CLI not found (npm install -g @anthropic-ai/claude-code)")
    cmd = [
        exe, "-p",
        "--model", model,
        "--output-format", "json",
        "--json-schema", json.dumps(schema),
        "--system-prompt", system,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"claude timed out after {timeout}s") from e
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise LLMError(f"claude exit {proc.returncode}: {proc.stderr[-500:] or proc.stdout[-500:]}") from e
    if out.get("is_error") or out.get("structured_output") is None:
        raise LLMError(f"claude failed: {out.get('subtype')} {str(out.get('result'))[:300]}")
    u = out.get("usage", {})
    log.info(
        "claude %s: in=%s cache_read=%s out=%s",
        model,
        u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
        u.get("cache_read_input_tokens", 0),
        u.get("output_tokens", 0),
    )
    return out["structured_output"]
