"""Provider-agnostic LLM client for the NL parser.

Transports, one interface:

``opencode``
    Drives the local **opencode** runtime (https://opencode.ai).  Three
    transports are tried in the order documented in the README:

    1. ``http``    - OpenAI-compatible ``POST {server}/v1/chat/completions``
    2. ``session`` - the opencode session API (``POST /session`` + a message)
    3. ``cli``     - ``opencode run --format json`` (last resort)

    Every transport degrades gracefully: a route the runtime does not expose
    (the v2 app server answers ``405``/HTML on ``/session`` and
    ``/v1/chat/completions``) is remembered as unavailable and the next
    transport is used.  The CLI transport is the one that is known to work
    with the local runtime, so it stays the last resort.

    The configured timeout is the budget of a **whole** call: the models and
    the transports are tried inside that budget and no single attempt may
    exceed what is left of it, so one parse can never block for the sum of
    all attempts.

    The CLI is pointed at the runtime server with ``--server`` only when the
    server was explicitly configured (``OPENCODE_BASE_URL`` set in the
    environment or ``OPENCODE_CLI_USE_SERVER=true``); with the default
    (unset) server URL the CLI uses its own runtime instead of a server that
    may not even be running.

``openai`` / ``openrouter`` / ``ollama`` / ``openai_compatible``
    Plain **OpenAI-compatible** ``POST {base_url}/chat/completions`` calls.
    ``openai_compatible`` (alias ``openai_compat``) is the escape hatch for
    any server that speaks the OpenAI chat-completions contract (vLLM, LM
    Studio, llama.cpp, TGI, DeepSeek, OpenRouter, ...).

The module has no LLM SDK dependency (no LangChain, no OpenAI SDK): it uses
the standard library plus loguru, which the backend already requires, so it
keeps working when LangChain is absent.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

ROOT = Path(__file__).resolve().parents[3]   # repo root, not backend/

# Model id used when neither the active config nor the environment names a
# usable opencode model.  A model id for the opencode runtime is always
# "provider/model"; anything else (a stale "gpt-4o-mini" in .env) is
# ignored.
DEFAULT_OPENCODE_MODEL = "opencode-go/beam-free"

# Transport order for the opencode provider.
DEFAULT_OPENCODE_TRANSPORTS: Tuple[str, ...] = ("http", "session", "cli")
VALID_TRANSPORTS = frozenset(DEFAULT_OPENCODE_TRANSPORTS)

# Placeholders the parser prompt must always carry.
PROMPT_PLACEHOLDERS: Tuple[str, ...] = ("query", "schema", "format_instructions", "reference_date")


class LLMError(RuntimeError):
    """Raised when no LLM backend could produce a usable reply."""


class LLMTransportUnavailable(LLMError):
    """The transport itself cannot be used (route missing, server down).

    Raised instead of :class:`LLMError` when the failure is not specific to
    a model, so the caller can skip the transport for the other models too.
    """


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|password|token)(\s*[:=]\s*)\S+"),
)


def redact_secrets(text: Any, extra: Iterable[str] = ()) -> str:
    """Return *text* with anything that looks like a credential masked.

    Used for every message that is raised, logged or stored, so an API key
    can never leak through an exception or a log line.
    """
    out = str(text)
    for secret in extra:
        secret = str(secret or "")
        if len(secret) >= 6:
            out = out.replace(secret, "***")
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("***", out)
    return out


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _balanced_objects(text: str) -> List[str]:
    """Return every balanced ``{...}`` block, ignoring braces in strings."""
    out: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out


def _balanced_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block, ignoring braces in strings."""
    objects = _balanced_objects(text)
    return objects[0] if objects else None


def _repair(blob: str) -> str:
    """Fix the small syntactic slips small models make when emitting JSON."""
    out = blob
    # ``"key\:"`` -> ``"key":`` and ``"key\"`` -> ``"key"`` (stray escapes).
    out = re.sub(r"\\([:\",])", r"\1", out)
    out = re.sub(r"\\n(?=\s*[}\]])", "", out)
    # Unquoted keys -> quoted keys.
    out = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', out)
    # Trailing commas before a closing bracket.
    out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Fields a parser reply is expected to carry.  A reply that only echoes the
# schema (or an example) has none of them, so it is not mistaken for the
# answer.
_FILTER_FIELDS = frozenset({
    "query_type", "semantic_query", "raw_query", "objects", "time",
    "count_constraint", "camera_id", "color", "action", "zone",
    "result_limit", "ask_color", "location_hint", "query_subtype",
})


def placeholders(template: str) -> List[str]:
    """Return the distinct ``{placeholder}`` names used by *template*."""
    names: List[str] = []
    for match in _PLACEHOLDER_RE.finditer(template or ""):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def render_prompt(template: str, *, required: Iterable[str] = (), **values: Any) -> str:
    """Substitute ``{placeholder}`` tokens, leaving unknown ones untouched.

    The parser prompt carries ``{query}``, ``{schema}``,
    ``{format_instructions}`` and ``{reference_date}``.  A missing value used
    to reach the model literally, which made it resolve relative dates
    against the session date instead of the archive epoch.  Names listed in
    *required* must therefore be present (and not ``None``); a missing one
    raises :class:`LLMError` instead of being silently forwarded.

    Only one substitution pass is made, so a value that itself contains a
    ``{...}`` token is never re-expanded, and literal braces in the template
    (e.g. an example JSON object) are left alone.
    """
    missing = [name for name in required if name not in values or values[name] is None]
    if missing:
        raise LLMError("prompt is missing values for: " + ", ".join(sorted(missing)))

    def _sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key in values:
            return str(values[key])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_sub, template)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model reply, tolerating fences and prose.

    Small models often echo an example (the prompt itself carries a JSON
    schema) before the real answer.  The *last* object is therefore the
    better guess, and an object that carries a known filter field wins over
    one that does not.

    Raises:
        LLMError: if no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise LLMError("empty LLM reply")
    blobs: List[str] = []
    for match in _FENCE.finditer(text):
        blobs.append(match.group(1).strip())
    blobs.append(text.strip())
    blobs.extend(_balanced_objects(text))

    parsed: List[Dict[str, Any]] = []
    seen: set = set()
    for blob in blobs:
        if not blob:
            continue
        for attempt in (blob, _repair(blob)):
            key = attempt.strip()
            if key in seen:
                continue
            seen.add(key)
            try:
                obj = json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                parsed.append(obj)
                break
    if not parsed:
        raise LLMError(f"no JSON object in LLM reply: {redact_secrets(text[:200])!r}")
    preferred = [obj for obj in parsed if _FILTER_FIELDS & set(obj.keys())]
    if preferred:
        return preferred[-1]
    return parsed[-1]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

# Routes that are "not available" rather than "broken": the opencode v2 app
# server answers 404/405/501 for the v1 routes and serves the SPA HTML.
_MISSING_ROUTE_STATUS = (404, 405, 501)


def _safe_url(url: str) -> str:
    """Strip the credentials from a URL so it can appear in a message."""
    return re.sub(r"(://)[^/@\s]+@", r"\1***@", str(url))


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: int,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """POST *payload* as JSON and return the decoded object.

    Raises:
        LLMTransportUnavailable: the route does not exist, the server is
            unreachable, the answer is not JSON (the SPA fallback) or the
            socket timed out - none of these is specific to a model, so the
            transport is remembered as unavailable.
        LLMError: any other failure (auth, quota, ...).
    """
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    safe = _safe_url(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        status = int(getattr(exc, "code", 0) or 0)
        if status in _MISSING_ROUTE_STATUS:
            raise LLMTransportUnavailable(f"HTTP {status}: route {safe} is not available") from exc
        raise LLMError(f"chat request to {safe} failed with HTTP {status}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        reason = getattr(exc, "reason", exc)
        raise LLMTransportUnavailable(f"cannot reach {safe}: {type(reason).__name__}") from exc
    except TimeoutError as exc:  # pragma: no cover - network path
        # A socket timeout says nothing about the model: the transport is
        # remembered as unavailable instead of being retried per model.
        raise LLMTransportUnavailable(
            f"chat request to {safe} timed out after {timeout}s"
        ) from exc
    except OSError as exc:  # pragma: no cover - network path
        raise LLMError(f"chat request to {safe} failed: {type(exc).__name__}") from exc

    if not raw:
        raise LLMTransportUnavailable(f"empty response from {safe}")
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # The local opencode app server answers unknown POST routes with the
        # SPA HTML: that means "route not available", not "model failed".
        raise LLMTransportUnavailable(f"non-JSON response from {safe}") from exc
    if not isinstance(data, dict):
        raise LLMTransportUnavailable(f"unexpected non-object response from {safe}")
    return data


def _http_delete(url: str, *, timeout: int, headers: Optional[Dict[str, str]] = None) -> None:
    """Best-effort DELETE used to release a session; errors are ignored."""
    request = urllib.request.Request(url, headers=headers or {}, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except Exception:
        pass


def _openai_payload(
    prompt: str,
    model: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    return payload


def _openai_content(data: Dict[str, Any]) -> str:
    """Extract the assistant text from an OpenAI-compatible response."""
    content: Any = None
    try:
        choices = data.get("choices") or []
        message = choices[0].get("message") or {}
        content = message.get("content")
    except (AttributeError, IndexError, KeyError, TypeError):
        content = None
    if isinstance(content, list):
        pieces = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        content = "".join(pieces)
    if content is None:
        error = data.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise LLMError(f"provider error: {redact_secrets(message)}")
        raise LLMError("chat response contained no message content")
    return str(content)


def _http_candidates(base_url: str) -> List[str]:
    """OpenAI-compatible routes to try on *base_url* (in order)."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return [f"{base}/chat/completions"]
    return [f"{base}/v1/chat/completions", f"{base}/chat/completions"]


def _split_model_id(model: str) -> Tuple[Optional[str], str]:
    """Split ``"provider/model"`` into its two parts."""
    text = str(model or "").strip()
    if "/" not in text:
        return None, text
    provider, _, name = text.partition("/")
    provider, name = provider.strip(), name.strip()
    if not provider or not name:
        return None, text
    return provider, name


def _session_text(payload: Dict[str, Any]) -> str:
    """Collect the text parts of an opencode session-API response."""
    chunks: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
            for key in ("parts", "content", "messages", "info"):
                if key in node:
                    _walk(node[key])
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    seen: List[str] = []
    for chunk in chunks:
        if chunk not in seen:
            seen.append(chunk)
    return "\n".join(seen).strip()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@dataclass
class LLMReply:
    text: str
    model: str
    transport: str
    raw: str = ""


class OpenAICompatBackend:
    """Any server that speaks the OpenAI ``/chat/completions`` contract."""

    transport = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = int(timeout)
        self.extra_headers = dict(extra_headers or {})

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
             max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        payload = _openai_payload(prompt, self.model, system=system, json_mode=json_mode,
                                  max_tokens=max_tokens, temperature=temperature)
        url = f"{self.base_url}/chat/completions"
        try:
            data = _http_post_json(url, payload, timeout=self.timeout, headers=self._headers())
        except LLMError as exc:
            raise LLMError(redact_secrets(str(exc), [self.api_key])) from exc
        text = _openai_content(data)
        raw = redact_secrets(json.dumps(data, ensure_ascii=False)[:2000], [self.api_key])
        return LLMReply(text=text, model=self.model, transport=self.transport, raw=raw)


# Transports that were found to be missing are remembered for a while so a
# runtime without the v1 routes is not probed again on every single parse.
_TRANSPORT_UNAVAILABLE_UNTIL: Dict[str, float] = {}


def reset_transport_cache() -> None:
    """Forget which transports were found to be unavailable."""
    _TRANSPORT_UNAVAILABLE_UNTIL.clear()


class OpencodeBackend:
    """Drives the local opencode runtime.

    ``model`` is the *primary* model id and is never mutated: a fallback
    model is used for one call only, the next call starts again with the
    configured primary.
    """

    transport = "opencode"

    def __init__(self, model: str, server: Optional[str] = None, cli: Optional[str] = None,
                 timeout: float = 600, cwd: Optional[Path] = None,
                 fallbacks: Optional[List[str]] = None,
                 transports: Optional[Sequence[str]] = None,
                 api_key: str = "", server_password: str = "",
                 transport_retry_seconds: int = 120,
                 cli_use_server: bool = False) -> None:
        self.model = str(model or "").strip()
        self.primary_model = self.model
        self.server = str(server or "").rstrip("/")
        # The timeout is the budget of a whole chat() call, not of a single
        # attempt, so a parse can never block for models x transports x
        # timeout.
        self.timeout = float(timeout) if timeout else 600.0
        self.cwd = cwd or ROOT
        self._cli = cli or None
        self.api_key = api_key or ""
        self.server_password = server_password or ""
        self.transport_retry_seconds = int(transport_retry_seconds)
        # The CLI is only pointed at a server that was explicitly configured.
        self.cli_use_server = bool(cli_use_server)
        self.transports = _normalise_transports(transports)
        cleaned: List[str] = []
        for candidate in (fallbacks or []):
            name = str(candidate or "").strip()
            if name and name != self.model and name not in cleaned:
                cleaned.append(name)
        self.fallbacks = cleaned

    # -- CLI discovery -----------------------------------------------------
    @property
    def cli(self) -> str:
        """Path of the opencode CLI (resolved lazily, never at build time)."""
        if not self._cli:
            self._cli = self._find_cli()
        return self._cli

    @cli.setter
    def cli(self, value: Optional[str]) -> None:
        self._cli = value or None

    @staticmethod
    def _find_cli() -> str:
        found = shutil.which("opencode")
        if found:
            return found
        for candidate in (
            Path.home() / "AppData/Roaming/npm/opencode.cmd",
            Path.home() / ".local/bin/opencode",
            Path("/usr/local/bin/opencode"),
        ):
            if candidate.exists():
                return str(candidate)
        raise LLMError(
            "opencode CLI not found; install opencode or set LLM_PROVIDER to an "
            "OpenAI-compatible endpoint"
        )

    # -- helpers ------------------------------------------------------------
    def _secrets(self) -> List[str]:
        return [self.api_key, self.server_password]

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.server_password:
            headers["Authorization"] = f"Bearer {self.server_password}"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def model_sequence(self) -> List[str]:
        """Primary model first, then the configured fallbacks."""
        sequence: List[str] = []
        if self.model:
            sequence.append(self.model)
        for candidate in self.fallbacks:
            if candidate and candidate not in sequence:
                sequence.append(candidate)
        return sequence

    def _transport_key(self, transport: str) -> str:
        return f"{transport}@{self.server}"

    def _transport_available(self, transport: str) -> bool:
        until = _TRANSPORT_UNAVAILABLE_UNTIL.get(self._transport_key(transport))
        if until is None:
            return True
        if time.monotonic() >= until:
            _TRANSPORT_UNAVAILABLE_UNTIL.pop(self._transport_key(transport), None)
            return True
        return False

    def _mark_unavailable(self, transport: str) -> None:
        window = max(0, int(self.transport_retry_seconds))
        _TRANSPORT_UNAVAILABLE_UNTIL[self._transport_key(transport)] = time.monotonic() + window

    # -- public API ---------------------------------------------------------
    def chat(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
             max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        """Send one prompt, trying the fallbacks and the transports in order.

        The configured primary model is always tried first.  A fallback is
        used only for the call that failed, the next call starts again with
        the primary.

        The whole call is bounded by ``self.timeout``: once the deadline has
        passed no further attempt is started, and every attempt gets at most
        the time that is left.  The CLI stays the last resort, but it can
        never exceed the remaining budget either.
        """
        deadline = time.monotonic() + self.timeout
        errors: List[str] = []
        dead_transports: set = set()
        for model in self.model_sequence():
            for transport in self.transports:
                if transport in dead_transports:
                    continue
                if not self._transport_available(transport):
                    errors.append(f"{transport}: unavailable (skipped)")
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMError(
                        f"opencode deadline of {self.timeout:g}s exceeded -> "
                        + " | ".join(errors)
                    )
                attempt_timeout = min(self.timeout, remaining)
                try:
                    return self._chat_once(prompt, model=model, transport=transport,
                                           timeout=attempt_timeout, deadline=deadline,
                                           system=system, json_mode=json_mode,
                                           max_tokens=max_tokens, temperature=temperature)
                except LLMTransportUnavailable as exc:
                    dead_transports.add(transport)
                    self._mark_unavailable(transport)
                    errors.append(f"{model}/{transport}: {redact_secrets(exc, self._secrets())}")
                except LLMError as exc:
                    errors.append(f"{model}/{transport}: {redact_secrets(exc, self._secrets())}")
        raise LLMError("all opencode transports failed -> " + " | ".join(errors))

    def _chat_once(self, prompt: str, model: str, transport: str,
                   timeout: Optional[float] = None, deadline: Optional[float] = None,
                   system: Optional[str] = None, json_mode: bool = False,
                   max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        """Send one prompt to the opencode runtime over *transport*."""
        budget = self.timeout if timeout is None else max(0.001, float(timeout))
        if transport == "http":
            return self._chat_http(prompt, model, timeout=budget, deadline=deadline,
                                   system=system, json_mode=json_mode,
                                   max_tokens=max_tokens, temperature=temperature)
        if transport == "session":
            return self._chat_session(prompt, model, timeout=budget, deadline=deadline,
                                      system=system, json_mode=json_mode,
                                      max_tokens=max_tokens, temperature=temperature)
        return self._chat_cli(prompt, model, timeout=budget, deadline=deadline,
                               system=system, json_mode=json_mode,
                               max_tokens=max_tokens, temperature=temperature)

    def _remaining(self, deadline: Optional[float], budget: float) -> float:
        """Return the time left for the next request of a transport."""
        if deadline is None:
            return budget
        return max(0.001, min(budget, deadline - time.monotonic()))

    # -- transport: OpenAI-compatible HTTP ----------------------------------
    def _chat_http(self, prompt: str, model: str, timeout: Optional[float] = None,
                   deadline: Optional[float] = None,
                   system: Optional[str] = None, json_mode: bool = False,
                   max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        budget = self.timeout if timeout is None else float(timeout)
        if not self.server:
            raise LLMTransportUnavailable("no opencode server URL configured (OPENCODE_BASE_URL)")
        payload = _openai_payload(prompt, model, system=system, json_mode=json_mode,
                                  max_tokens=max_tokens, temperature=temperature)
        headers = self._auth_headers()
        last: Optional[Exception] = None
        for url in _http_candidates(self.server):
            try:
                data = _http_post_json(url, payload, timeout=self._remaining(deadline, budget),
                                       headers=headers)
            except LLMTransportUnavailable as exc:
                last = exc
                continue
            text = _openai_content(data)
            raw = redact_secrets(json.dumps(data, ensure_ascii=False)[:2000], self._secrets())
            return LLMReply(text=text, model=model, transport="opencode-http", raw=raw)
        raise LLMTransportUnavailable(
            f"no OpenAI-compatible route on {_safe_url(self.server)} ({last})"
        )

    # -- transport: opencode session API -----------------------------------
    def _chat_session(self, prompt: str, model: str, timeout: Optional[float] = None,
                      deadline: Optional[float] = None,
                      system: Optional[str] = None, json_mode: bool = False,
                      max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        budget = self.timeout if timeout is None else float(timeout)
        if not self.server:
            raise LLMTransportUnavailable("no opencode server URL configured (OPENCODE_BASE_URL)")
        provider_id, model_id = _split_model_id(model)
        headers = self._auth_headers()
        session = _http_post_json(f"{self.server}/session", {"title": "nl-parser"},
                                  timeout=self._remaining(deadline, budget), headers=headers)
        session_id = str(session.get("id") or "").strip()
        if not session_id:
            raise LLMTransportUnavailable("opencode session API returned no session id")
        body: Dict[str, Any] = {
            "providerID": provider_id,
            "modelID": model_id,
            "parts": [{"type": "text", "text": prompt}],
        }
        if system:
            body["system"] = system
        try:
            response = _http_post_json(f"{self.server}/session/{session_id}/message", body,
                                       timeout=self._remaining(deadline, budget), headers=headers)
        finally:
            _http_delete(f"{self.server}/session/{session_id}",
                         timeout=self._remaining(deadline, budget), headers=headers)
        text = _session_text(response)
        if not text:
            raise LLMError("opencode session API returned no text")
        raw = redact_secrets(json.dumps(response, ensure_ascii=False)[:2000], self._secrets())
        return LLMReply(text=text, model=model, transport="opencode-session", raw=raw)

    # -- transport: opencode CLI (last resort) ------------------------------
    def _chat_cli(self, prompt: str, model: str, timeout: Optional[float] = None,
                  deadline: Optional[float] = None,
                  system: Optional[str] = None, json_mode: bool = False,
                  max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMReply:
        """Send one prompt to the opencode runtime.

        The parser prompt (instructions + JSON schema + query) is longer than
        the Windows command-line limit, so it is handed to the runtime as an
        attached file and the message asks for the JSON object it describes.
        The temporary file is always removed, also when the call fails.

        ``--server`` is only added when the runtime server was explicitly
        configured: pointing the CLI at a server that may not be running
        would make the transport fail for no reason.  The call itself never
        gets more time than the budget that is left.
        """
        budget = self.timeout if timeout is None else float(timeout)
        try:
            cli = self.cli
        except LLMError as exc:
            raise LLMTransportUnavailable(str(exc)) from exc

        cmd = [cli, "run"]
        if self.server and self.cli_use_server:
            cmd += ["--server", self.server]
        cmd += ["-m", model, "--format", "json"]
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".md", prefix="nl_prompt_",
                                             delete=False, encoding="utf-8") as handle:
                handle.write(prompt)
                tmp_path = Path(handle.name)
            cmd += ["--file", str(tmp_path)]
            cmd.append(
                "The attached file contains the parser instructions, the JSON schema "
                "and the user query. Reply with ONLY the JSON object for that query: "
                "no prose, no markdown fences, no explanation."
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._remaining(deadline, budget),
                    cwd=str(self.cwd),
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMError(
                    f"opencode run timed out after {self._remaining(deadline, budget):g}s"
                ) from exc
            except OSError as exc:
                raise LLMError(f"opencode run failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        text = self._cli_text(proc)
        if not text:
            tail = (proc.stderr or "")[-300:] or (proc.stdout or "")[-300:]
            raise LLMError(
                f"opencode produced no text (exit {proc.returncode}): "
                f"{redact_secrets(tail, self._secrets())}"
            )
        return LLMReply(text=text, model=model, transport="opencode-cli",
                         raw=redact_secrets((proc.stdout or "")[-2000:], self._secrets()))

    @staticmethod
    def _cli_text(proc: Any) -> str:
        """Read the assistant text out of the opencode NDJSON event stream."""
        texts: List[str] = []
        for line in (getattr(proc, "stdout", "") or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "error":
                error = event.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                raise LLMError(f"opencode error: {redact_secrets(message or event)}")
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            candidate = part.get("text") or event.get("text") or event.get("content")
            if isinstance(candidate, str) and candidate.strip():
                texts.append(candidate)
            elif isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        texts.append(item["text"])
        if texts:
            return "\n".join(texts).strip()
        # Some builds print a single JSON document instead of an event stream.
        try:
            document = json.loads((getattr(proc, "stdout", "") or "").strip() or "{}")
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(document, dict):
            return ""
        for key in ("text", "content", "output", "response", "result"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested = value.get("text") or value.get("content")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
        return ""


def _normalise_transports(transports: Optional[Sequence[str]]) -> List[str]:
    """Normalise the configured transport order (``auto`` expands to all)."""
    if not transports:
        return list(DEFAULT_OPENCODE_TRANSPORTS)
    if isinstance(transports, str):
        transports = [part for part in re.split(r"[,\s]+", transports) if part]
    ordered: List[str] = []
    for item in transports:
        name = str(item or "").strip().lower()
        if not name:
            continue
        if name == "auto":
            for candidate in DEFAULT_OPENCODE_TRANSPORTS:
                if candidate not in ordered:
                    ordered.append(candidate)
        elif name in VALID_TRANSPORTS and name not in ordered:
            ordered.append(name)
    return ordered or list(DEFAULT_OPENCODE_TRANSPORTS)


def _opencode_model_id(value: Any) -> Optional[str]:
    """Return *value* when it is a usable ``provider/model`` id, else ``None``."""
    text = str(value or "").strip()
    if not text or "/" not in text:
        return None
    provider, _, model = text.partition("/")
    if not provider.strip() or not model.strip():
        return None
    if any(ch.isspace() for ch in text):
        return None
    return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend(cfg: Optional[Dict[str, Any]] = None) -> Any:
    """Create the LLM backend described by the active configuration.

    The ``opencode`` provider never raises here: a stale model id, a missing
    CLI or an unreachable runtime are handled by the backend itself (the CLI
    is resolved lazily and the transports degrade gracefully).
    """
    from backend.app.config import settings

    llm_cfg = cfg or settings.get_active_llm_config()
    provider = str(llm_cfg.get("provider") or "").strip().lower()
    model = str(llm_cfg.get("model") or "").strip()
    api_key = str(llm_cfg.get("api_key") or "").strip()

    timeout = int(getattr(settings, "LLM_TIMEOUT", 0) or settings.NL_TIMEOUT_SECONDS)

    if provider == "opencode":
        # A model id for the opencode runtime is always "provider/model".
        # Anything else (e.g. a stale "gpt-4o-mini" in .env) is ignored in
        # favour of the configured opencode model.
        primary = (
            _opencode_model_id(model)
            or _opencode_model_id(settings.OPENCODE_MODEL)
            or DEFAULT_OPENCODE_MODEL
        )
        fallbacks: List[str] = []
        for raw in str(settings.OPENCODE_MODEL_FALLBACKS or "").split(","):
            candidate = _opencode_model_id(raw.strip())
            if candidate and candidate != primary and candidate not in fallbacks:
                fallbacks.append(candidate)
        # The CLI only talks to a runtime server that was asked for.
        cli_use_server = bool(
            getattr(settings, "OPENCODE_CLI_USE_SERVER", False)
            or getattr(settings, "OPENCODE_BASE_URL_EXPLICIT", False)
        )
        return OpencodeBackend(
            model=primary,
            server=settings.OPENCODE_BASE_URL or None,
            cli=settings.OPENCODE_CLI or None,
            timeout=timeout,
            fallbacks=fallbacks,
            transports=settings.OPENCODE_TRANSPORT,
            server_password=settings.OPENCODE_SERVER_PASSWORD,
            transport_retry_seconds=int(getattr(settings, "OPENCODE_TRANSPORT_RETRY_SECONDS", 120)),
            cli_use_server=cli_use_server,
        )
    if provider in ("openai_compatible", "openai_compat", "vllm", "lmstudio",
                    "llamacpp", "tgi"):
        base = settings.OPENAI_BASE_URL or llm_cfg.get("base_url") or ""
        if not base:
            raise LLMError("OPENAI_BASE_URL is required for an OpenAI-compatible endpoint")
        if not model:
            raise LLMError(
                f"a model id is required for the '{provider}' provider "
                "(set NL_DEFAULT_MODEL or the model in the llm_config document)"
            )
        return OpenAICompatBackend(base, api_key, model, timeout=timeout)
    if provider in ("openai", "openrouter", "ollama", ""):
        base = settings.OPENAI_BASE_URL or (
            settings.OPENROUTER_BASE_URL if provider == "openrouter"
            else "https://api.openai.com/v1"
        )
        if provider == "ollama":
            base = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"
        key = api_key or settings.OPENAI_API_KEY
        if provider == "openai" and not key and not settings.OPENAI_BASE_URL:
            raise LLMError("OPENAI_API_KEY is not set")
        # Every provider has its own default model: the opencode model id
        # must never be sent to the hosted API.  Without a provider the
        # LangChain path decides (OpenAI when a key is set, else Ollama).
        if provider:
            default_model = settings.default_model_for(provider)
        else:
            default_model = (
                settings.default_model_for("openai") if key
                else settings.default_model_for("ollama")
            )
        return OpenAICompatBackend(base, key, model or default_model, timeout=timeout)
    raise LLMError(f"Unsupported LLM provider: '{provider}'")


def chat(
    prompt: str,
    *,
    system: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    backend: Optional[Any] = None,
) -> LLMReply:
    """Single entry point used by the parser and the evaluation scripts."""
    backend = backend or build_backend()
    return backend.chat(prompt, system=system, json_mode=json_mode,
                        max_tokens=max_tokens, temperature=temperature)


def chat_json(prompt: str, *, system: Optional[str] = None, backend: Optional[Any] = None,
              max_tokens: Optional[int] = None, temperature: float = 0.0) -> Dict[str, Any]:
    """Chat and decode a JSON object from the reply."""
    reply = chat(prompt, system=system, json_mode=True, backend=backend,
                 max_tokens=max_tokens, temperature=temperature)
    return extract_json_object(reply.text)


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

class ParseCache:
    """JSON-lines cache so a parse run is reproducible and cheap to repeat.

    A cache file that cannot be read is reported (it means every query has
    to be parsed - and paid for - again) and the cache then starts empty.
    Writes are atomic: the data goes to a temporary file in the same
    directory and is then moved over the target, so a crash in the middle of
    a write cannot leave a truncated cache behind.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Parse cache {} could not be read ({}); the queries in it "
                    "have to be parsed again.",
                    self.path, self.load_error,
                )
                self._data = {}
            else:
                self._data = loaded if isinstance(loaded, dict) else {}

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp_path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


def default_cache_path(model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return ROOT / "backend" / "benchmarks" / "parsed_cache" / f"{safe}.json"
